"""Discovery-only head clustering with prompt-resampled stability."""
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
from tqdm import tqdm
from src.features.rows import BOS_MODES, METRICS
from src.features.task_score import TASK_COLUMNS
from src.utils.cache import file_hash, fingerprint, write_csv, write_npz
from src.utils.io import write_json
from src.utils.logging import save_metadata

GROUPS = {'entropy': ['entropy_normalized'], 'distance': ['mean_distance'],
          'sparsity': ['gini', 'top1_mass'], 'task': ['best_task_score']}
COLUMNS = [column for group in GROUPS.values() for column in group]
RAW_COLUMNS = COLUMNS[:4] + ['task_score']
OUTPUTS = ('assignments.csv', 'cluster_profiles.csv', 'layer_distribution.csv',
           'pca_coordinates.csv', 'pca_loadings.csv', 'k_selection.csv',
           'bootstrap_stability.csv', 'stability_summary.csv', 'shortlist.csv',
           'fit_parameters.npz', 'selection_rules.json', 'metadata.json')


def specialize(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute task z scores across heads and their maximum; preserve missingness."""
    z = np.full_like(scores, np.nan, dtype=float)
    for task in range(scores.shape[1]):
        valid = np.isfinite(scores[:, task])
        if valid.any():
            values = scores[valid, task]
            mean, std = values.mean(), values.std(ddof=0)
            z[valid, task] = (values-mean)/std if std > np.finfo(float).eps * max(1, abs(mean)) else 0
    best = np.where(np.isfinite(z), z, -np.inf).max(axis=1)
    best[~np.isfinite(z).any(axis=1)] = np.nan
    return best, z


def standardized_space(frame: pd.DataFrame, settings: dict) -> tuple[np.ndarray, StandardScaler, np.ndarray]:
    """Standardize discovery features; give each of four feature groups equal weight."""
    values = frame[COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('Clustering features contain undefined values; inspect BOS-mode valid counts')
    weights = settings.get('group_weights', dict.fromkeys(GROUPS, 1.0))
    if set(weights) != set(GROUPS) or any(not np.isfinite(w) or w <= 0 for w in weights.values()):
        raise ValueError('group_weights must give a positive finite weight to each of four groups')
    factors = np.array([np.sqrt(weights[group]/len(columns)) for group, columns in GROUPS.items() for _ in columns])
    scaler = StandardScaler().fit(values)
    return scaler.transform(values)*factors, scaler, factors


def fit_partition(space: np.ndarray, method: str, k: int, settings: dict, seed: int):
    """Fit a seeded KMeans or regularized Gaussian mixture partition."""
    if method == 'kmeans':
        estimator = KMeans(n_clusters=k, random_state=seed, n_init=settings.get('n_init', 10), max_iter=settings.get('max_iter', 500))
    elif method == 'gmm':
        estimator = GaussianMixture(n_components=k, random_state=seed, n_init=settings.get('gmm_n_init', 3),
            reg_covar=settings.get('gmm_reg_covar', 1e-6), max_iter=settings.get('max_iter', 500))
    else:
        raise ValueError(f'Unknown clustering method: {method}')
    labels = estimator.fit_predict(space)
    if method == 'gmm' and not estimator.converged_:
        raise RuntimeError(f'Gaussian mixture did not converge for k={k}; adjust max_iter/reg_covar')
    return labels, estimator


def choose_k(space: np.ndarray, method: str, settings: dict, seed: int) -> tuple[np.ndarray, object, list[dict]]:
    """Choose the highest silhouette over valid k; deterministic ties prefer smaller k."""
    candidates = sorted(set(settings['k_range']))
    if any(not isinstance(k, int) or k < 2 for k in candidates):
        raise ValueError('k_range must contain integers >= 2')
    rows, fitted = [], {}
    unique = len(np.unique(space, axis=0))
    for k in candidates:
        row = {'method': method, 'k': k, 'silhouette': np.nan, 'status': 'invalid_size', 'selected': False}
        if 2 <= k < len(space) and k <= unique:
            labels, estimator = fit_partition(space, method, k, settings, seed)
            if 1 < len(np.unique(labels)) < len(space):
                row.update(silhouette=float(silhouette_score(space, labels)), status='ok')
                fitted[k] = (labels, estimator)
            else:
                row['status'] = 'collapsed_partition'
        rows.append(row)
    valid = [row for row in rows if row['status']=='ok']
    if not valid:
        raise ValueError(f'No valid {method} silhouette candidate; need >= 3 distinct heads and k < head count')
    best = max(valid, key=lambda row: (row['silhouette'], -row['k']))
    best['selected'] = True
    labels, estimator = fitted[best['k']]
    return labels, estimator, rows


def candidate_shortlist(assignments: pd.DataFrame, settings: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select discovery candidates from task-aligned cluster profiles, never validate them."""
    rules = settings.get('selection', {})
    threshold = rules.get('min_cluster_task_z', 0.5)
    minimum = rules.get('min_head_specialization', 1.0)
    agreement = rules.get('min_task_agreement', 0.5)
    top = settings.get('shortlist_per_cluster', 3)
    if not np.isfinite([threshold, minimum, agreement]).all() or not 0 <= agreement <= 1 or not isinstance(top,int) or top < 1:
        raise ValueError('Invalid shortlist thresholds/count')
    tasks = list(TASK_COLUMNS)
    z_columns = ['z_'+label for label in TASK_COLUMNS.values()]
    profiles, shortlist = [], []
    for cluster, group in assignments.groupby('cluster', sort=True):
        means = group[z_columns].mean().to_numpy()
        dominant_index = int(means.argmax())
        dominant = tasks[dominant_index]
        fraction = float((group['best_task']==dominant).mean())
        aligned = bool(means[dominant_index] >= threshold and fraction >= agreement)
        profile = {'cluster': cluster, 'n_heads': len(group), 'dominant_task': dominant,
                   'dominant_task_mean_z': float(means[dominant_index]), 'task_agreement': fraction,
                   'task_aligned': aligned}
        for column in [*COLUMNS, *TASK_COLUMNS.values(), *z_columns, 'specialization']:
            profile['mean_'+column] = float(group[column].mean())
        profiles.append(profile)
        if aligned:
            eligible = group[(group['specialization'] >= minimum) & (group['best_task']==dominant)]
            eligible = eligible.sort_values(['specialization','layer','head'], ascending=[False,True,True]).head(top)
            for rank, (_, head) in enumerate(eligible.iterrows(), 1):
                shortlist.append({'layer': int(head['layer']), 'head': int(head['head']), 'cluster': int(cluster),
                    'task': dominant, 'specialization': head['specialization'], 'rank_in_cluster': rank,
                    'status': 'discovery_candidate', 'causally_validated': False})
    columns = ['layer','head','cluster','task','specialization','rank_in_cluster','status','causally_validated']
    return pd.DataFrame(profiles), pd.DataFrame(shortlist, columns=columns)


def prompt_arrays(index: dict, attention: Path, work: Path, mode: str, heads: pd.DataFrame) -> tuple[dict, dict]:
    """Stream discovery summary columns into numeric on-disk arrays for bootstrap draws."""
    work.mkdir(parents=True, exist_ok=True)
    mode_index = BOS_MODES.index(mode)
    slots = [METRICS.index(column) for column in RAW_COLUMNS]
    layers = heads['layer'].to_numpy(dtype=int)
    head_indices = heads['head'].to_numpy(dtype=int)
    arrays, strata = {}, {}
    selected = sorted([entry for entry in index['records'].values() if entry['split']=='discovery'], key=lambda e:e['id'])
    signature = fingerprint({'bos_mode':mode},[Path(__file__)],extra={
        'entries':selected,'heads':heads[['layer','head']].values.tolist(),'architecture':index['architecture']})
    manifest_path = work/'manifest.json'
    prior = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    cached = prior.get('files',{}) if prior.get('signature')==signature else {}
    manifest = {'signature':signature,'files':{}}
    for task in TASK_COLUMNS:
        entries = sorted([entry for entry in index['records'].values() if entry['split']=='discovery' and entry['task']==task], key=lambda e:e['id'])
        if not entries:
            raise ValueError(f'Missing discovery prompt summaries for {task}')
        path = work/(task+'.npy')
        expected_shape = (len(entries),len(heads),len(RAW_COLUMNS))
        if task in cached and path.exists() and file_hash(path)==cached[task]:
            array = np.load(path,mmap_mode='r',allow_pickle=False)
            if array.shape!=expected_shape or array.dtype!=np.dtype('float64'):
                raise ValueError(f'Unexpected cached bootstrap array shape/dtype: {path}')
            arrays[task] = array
            strata[task] = np.array([entry['template_id'] for entry in entries])
            manifest['files'][task] = cached[task]
            continue
        array = np.lib.format.open_memmap(path, mode='w+', dtype='float64', shape=expected_shape)
        for i, entry in enumerate(tqdm(entries, desc=f'Bootstrap input {task}')):
            with np.load(attention/entry['file'], allow_pickle=False) as shard:
                summary = shard['selected_summary']
                expected = (2,index['architecture']['n_layers'],index['architecture']['n_heads'],len(METRICS))
                if summary.shape != expected:
                    raise ValueError(f'Unexpected summary dimensions for {entry["id"]}')
                array[i] = summary[mode_index,layers,head_indices][:,slots]
        array.flush()
        manifest['files'][task] = file_hash(path)
        write_json(manifest_path,manifest)
        arrays[task] = array
        strata[task] = np.array([entry['template_id'] for entry in entries])
    write_json(manifest_path,manifest)
    return arrays, strata


def resampled_frame(arrays: dict, strata: dict, heads: pd.DataFrame, rng: np.random.Generator | None,
                    stratified: bool = True) -> pd.DataFrame:
    """Resample prompt indices jointly across heads within each task/template."""
    result = heads.copy()
    general, scores = [], []
    for task, array in arrays.items():
        weights = np.ones(len(array), dtype=int)
        if rng is not None:
            weights[:] = 0
            groups = np.unique(strata[task]) if stratified else [None]
            for template in groups:
                indices = np.arange(len(array)) if template is None else np.flatnonzero(strata[task]==template)
                draw = rng.choice(indices, size=len(indices), replace=True)
                weights += np.bincount(draw, minlength=len(array))
        # Accumulate blocks to bound scratch memory independently of prompt count.
        sums = np.zeros(array.shape[1:]); counts = np.zeros(array.shape[1:])
        for start in range(0,len(array),128):
            block = np.asarray(array[start:start+128])
            finite = np.isfinite(block)
            weight = weights[start:start+128,None,None]
            sums += (np.where(finite,block,0)*weight).sum(axis=0)
            counts += (finite*weight).sum(axis=0)
        means = np.divide(sums,counts,out=np.full_like(sums,np.nan),where=counts>0)
        general.append(means[:,:4]); scores.append(means[:,4])
    general = np.mean(general,axis=0)
    for i, column in enumerate(COLUMNS[:4]):
        result[column] = general[:,i]
    scores = np.stack(scores,axis=1)
    specialization, z = specialize(scores)
    result['best_task_score'] = result['specialization'] = specialization
    for i, label in enumerate(TASK_COLUMNS.values()):
        result[label] = scores[:,i]
        result['z_'+label] = z[:,i]
    return result


def cluster_heads(features: Path, attention: Path, folder: Path, config: dict) -> dict[str, Path]:
    """Fit/save discovery partitions, PCA, profiles, shortlist and prompt-bootstrap ARI."""
    features, attention, folder = Path(features), Path(attention), Path(folder)
    inputs = [features/'head_features.csv', features/'completion.json', features/'metadata.json', attention/'index.json']
    for path in inputs:
        if not path.exists():
            raise FileNotFoundError(f'Missing clustering input: {path}; run steps 02 and 03')
    feature_completion = json.loads(inputs[1].read_text())
    if file_hash(inputs[0]) != feature_completion['hashes']['head_features.csv']:
        raise ValueError('Head feature checksum mismatch; rerun step 03')
    index = json.loads(inputs[3].read_text())
    if not index['complete'] or feature_completion['extraction_signature'] != index['signature']:
        raise ValueError('Features and extraction do not describe the same completed run')
    settings = config['clustering']
    mode = settings.get('bos_mode','excluded')
    if mode not in BOS_MODES:
        raise ValueError('clustering.bos_mode must be included or excluded')
    discovery_entries = [entry for entry in index['records'].values() if entry['split']=='discovery']
    for entry in tqdm(discovery_entries,desc='Verify discovery summaries'):
        path = attention/entry['file']
        if not path.exists():
            raise FileNotFoundError(f'Missing discovery shard: {path}; resume step 02')
        if file_hash(path) != entry['sha256']:
            raise ValueError(f'Discovery shard checksum mismatch: {path}; resume step 02')
    code = [Path(__file__), Path(__file__).resolve().parents[1]/'utils/cache.py']
    signature = fingerprint(config,[*inputs,*code],extra={'stage':'clustering'})
    completion = folder/'completion.json'
    if completion.exists():
        cached = json.loads(completion.read_text())
        if cached['signature']==signature and set(cached['hashes'])==set(OUTPUTS) and all(
            (folder/name).exists() and file_hash(folder/name)==checksum for name,checksum in cached['hashes'].items()):
            return {name:folder/name for name in OUTPUTS}
    table = pd.read_csv(inputs[0])
    heads = table[(table['split']=='discovery') & (table['bos_mode']==mode)].sort_values(['layer','head']).reset_index(drop=True)
    expected = {(layer,head) for layer in range(index['architecture']['n_layers']) for head in range(index['architecture']['n_heads'])}
    if len(heads)!=len(expected) or set(zip(heads['layer'],heads['head']))!=expected:
        raise ValueError('Discovery feature table must contain each model head exactly once')
    if not np.isfinite(heads[[*COLUMNS,*TASK_COLUMNS.values()]].to_numpy(dtype=float)).all():
        raise ValueError('Undefined discovery feature/task scores; inspect valid counts or choose another BOS mode')
    heads['split'], heads['bos_mode'] = 'discovery', mode
    space, scaler, factors = standardized_space(heads,settings)
    methods = ('kmeans','gmm')
    fits, selection = {}, []
    for method in methods:
        labels,estimator,rows = choose_k(space,method,settings,config['seed'])
        fits[method] = (labels,estimator)
        selection.extend(rows)
    primary = settings.get('primary_method','kmeans')
    if primary not in methods:
        raise ValueError('primary_method must be kmeans or gmm')
    heads['cluster'] = fits[primary][0]
    heads['kmeans_cluster'],heads['gmm_cluster'] = fits['kmeans'][0],fits['gmm'][0]
    pca = PCA(n_components=2,svd_solver='full').fit(space)
    coordinates = pca.transform(space)
    profiles, shortlist = candidate_shortlist(heads,settings)
    for table in (profiles,shortlist):
        table['split'],table['bos_mode'],table['method'] = 'discovery',mode,primary
    all_pairs = pd.MultiIndex.from_product([range(index['architecture']['n_layers']),sorted(heads['cluster'].unique())],names=['layer','cluster'])
    layer_distribution = heads.groupby(['layer','cluster']).size().reindex(all_pairs,fill_value=0).rename('n_heads').reset_index()
    layer_distribution['fraction_in_layer'] = layer_distribution['n_heads']/index['architecture']['n_heads']
    pca_frame = heads[['layer','head','cluster','kmeans_cluster','gmm_cluster','split','bos_mode']].copy()
    pca_frame['pc1'],pca_frame['pc2'] = coordinates[:,0],coordinates[:,1]
    loadings = pd.DataFrame({'feature':COLUMNS,'pc1_component':pca.components_[0], 'pc2_component':pca.components_[1],
        'pc1_loading':pca.components_[0]*np.sqrt(pca.explained_variance_[0]),
        'pc2_loading':pca.components_[1]*np.sqrt(pca.explained_variance_[1]),'group_factor':factors})
    repeats = settings['bootstrap_repeats']
    if not isinstance(repeats,int) or repeats<2:
        raise ValueError('bootstrap_repeats must be an integer >= 2')
    folder.mkdir(parents=True,exist_ok=True)
    arrays,strata = prompt_arrays(index,attention,folder/'bootstrap_inputs',mode,heads)
    rebuilt = resampled_frame(arrays,strata,heads,None)
    if not np.allclose(rebuilt[COLUMNS],heads[COLUMNS],rtol=1e-5,atol=1e-7,equal_nan=True):
        raise ValueError('Saved head features disagree with discovery prompt summaries; rerun step 03')
    boot_rows = []
    checkpoints = folder/'bootstrap_checkpoints'
    checkpoints.mkdir(exist_ok=True)
    for repeat in tqdm(range(repeats),desc='Prompt bootstrap clustering'):
        path = checkpoints/f'{repeat:05d}.json'
        if path.exists():
            saved = json.loads(path.read_text())
            payload_hash = hashlib.sha256(json.dumps(saved.get('rows'),sort_keys=True).encode()).hexdigest()
            if saved.get('signature')==signature and saved.get('sha256')==payload_hash and len(saved['rows'])==2 and {r['method'] for r in saved['rows']}==set(methods) and all(
                r['repeat']==repeat and np.isfinite(r['adjusted_rand_index']) and -1 <= r['adjusted_rand_index'] <= 1 for r in saved['rows']):
                boot_rows.extend(saved['rows'])
                continue
        rng = np.random.default_rng(np.random.SeedSequence([config['seed'],repeat]))
        sampled = resampled_frame(arrays,strata,heads,rng,settings.get('bootstrap_stratify_templates',True))
        sampled_space,_,_ = standardized_space(sampled,settings)
        rows = []
        for method in methods:
            base,estimator = fits[method]
            k = estimator.n_clusters if method=='kmeans' else estimator.n_components
            labels,_ = fit_partition(sampled_space,method,k,settings,config['seed'])
            rows.append({'repeat':repeat,'method':method,'k':k,'adjusted_rand_index':float(adjusted_rand_score(base,labels)),
                         'realized_clusters':len(np.unique(labels))})
        write_json(path,{'signature':signature,'rows':rows,'sha256':hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()})
        boot_rows.extend(rows)
    bootstrap = pd.DataFrame(boot_rows)
    stability = []
    for method,group in bootstrap.groupby('method',sort=True):
        values = group['adjusted_rand_index'].to_numpy()
        low,high = np.quantile(values,[0.025,0.975])
        stability.append({'method':method,'k':int(group['k'].iloc[0]),'repeats':len(values),
                          'mean_ari':values.mean(),'std_ari':values.std(ddof=1),'ari_p025':low,'ari_p975':high})
    tables = {'assignments.csv':heads,'cluster_profiles.csv':profiles,'layer_distribution.csv':layer_distribution,
              'pca_coordinates.csv':pca_frame,'pca_loadings.csv':loadings,'k_selection.csv':pd.DataFrame(selection),
              'bootstrap_stability.csv':bootstrap,'stability_summary.csv':pd.DataFrame(stability),'shortlist.csv':shortlist}
    for name,frame in tables.items():
        write_csv(folder/name,frame)
    parameters = {'feature_names':np.array(COLUMNS),'scaler_mean':scaler.mean_,'scaler_scale':scaler.scale_,
        'group_factors':factors,'pca_mean':pca.mean_,'pca_components':pca.components_,
        'pca_explained_variance':pca.explained_variance_,'pca_explained_variance_ratio':pca.explained_variance_ratio_,
        'kmeans_centers':fits['kmeans'][1].cluster_centers_,'gmm_means':fits['gmm'][1].means_,
        'gmm_covariances':fits['gmm'][1].covariances_,'gmm_weights':fits['gmm'][1].weights_}
    write_npz(folder/'fit_parameters.npz',**parameters)
    write_json(folder/'selection_rules.json',{'primary_method':primary,'bos_mode':mode,'feature_groups':GROUPS,
        'settings':settings,'status':'discovery_candidates_only','split':'discovery',
        'bootstrap':'Prompt resampling within task and optionally template, shared indices across heads; refit standardization, specialization and clustering at fixed discovery-selected k',
        'ari_interval':'Empirical percentile range across prompt-bootstrap partitions, not a confidence interval on mean ARI',
        'pca_loadings':'Components times sqrt(explained variance) in weighted standardized feature space'})
    metadata = json.loads((features/'metadata.json').read_text())
    save_metadata(folder,config,metadata['model_config'])
    write_json(completion,{'signature':signature,'hashes':{name:file_hash(folder/name) for name in OUTPUTS}})
    return {name:folder/name for name in OUTPUTS}
