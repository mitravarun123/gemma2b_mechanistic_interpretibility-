"""Patch all heads in both directions and assess discovery candidates causally."""
import argparse
import json
import logging
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.model.loader import load_model
from src.validation.patching import run_patching,aligned_twins
from src.validation.assessment import assess_candidates
from src.validation.common import read_validation_inputs,validate_model_inputs,stage_signature,cached_stage,behavioral_records
from src.utils.cache import file_hash
from src.utils.io import read_config,write_json
from src.utils.logging import configure_logging
from src.utils.seed import seed_all


def main() -> None:
    """Run all-head held-out patches and save feature/causal correlations and candidate status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/default.yaml')
    args = parser.parse_args(); config = read_config(args.config)
    configure_logging(); seed_all(config['seed'])
    root = Path(__file__).resolve().parents[1]
    discovery,records,twins,shortlist,paths = read_validation_inputs(root,config)
    results = root/config['results_dir']; folder = results/'patching'
    for parent,name in ((results/'ablation','summary.csv'),(results/'consistency','consistency.csv')):
        for path in (parent/name,parent/'completion.json'):
            if not path.exists():
                raise FileNotFoundError(f'Missing step 05 input: {path}; run 05_run_ablation.py')
        completion = json.loads((parent/'completion.json').read_text())
        if file_hash(parent/name)!=completion['hashes'].get(name):
            raise ValueError(f'Step 05 artifact checksum mismatch: {parent/name}')
        paths.extend([parent/name,parent/'completion.json'])
    if (folder/'metadata.json').exists():
        model_config = json.loads((folder/'metadata.json').read_text())['model_config']
        evaluation = behavioral_records(records,'validation')
        signature = stage_signature('patching',config,paths,{'model_config':model_config,
            'evaluation':evaluation,'twins':aligned_twins(evaluation,twins)})
        if cached_stage(folder,signature) and (folder/'validated_heads.csv').exists():
            logging.info('Completed patching and assessment cache verified; skipping model load')
            return
    model,tokenizer,_ = load_model(config)
    validate_model_inputs(model,tokenizer,[*records,*twins],results/'attention/metadata.json')
    output = run_patching(model,records,twins,folder,config,paths,results/'features/head_features.csv')
    assess_candidates(shortlist,results/'consistency/consistency.csv',results/'ablation/summary.csv',output,folder,config)
    completion = json.loads((folder/'completion.json').read_text())
    for name in ('validated_heads.csv','assessment_rules.json'):
        completion['hashes'][name] = file_hash(folder/name)
    write_json(folder/'completion.json',completion)
    logging.info('Patching summary: %s',output)


if __name__=='__main__':
    main()
