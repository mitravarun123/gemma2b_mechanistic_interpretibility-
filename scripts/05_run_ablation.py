"""Run held-out consistency and head/group ablation with matched random controls."""
import argparse
import json
import logging
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.model.loader import load_model
from src.validation.ablation import run_ablation
from src.validation.consistency import run_consistency
from src.validation.common import read_validation_inputs,validate_model_inputs,stage_signature,cached_stage,behavioral_records
from src.utils.io import read_config
from src.utils.logging import configure_logging
from src.utils.seed import seed_all


def main() -> None:
    """Load saved discovery choices and evaluate exclusively on held-out prompts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/default.yaml')
    args = parser.parse_args(); config = read_config(args.config)
    configure_logging(); seed_all(config['seed'])
    root = Path(__file__).resolve().parents[1]
    discovery,records,twins,shortlist,paths = read_validation_inputs(root,config)
    results = root/config['results_dir']
    run_consistency(results/'attention',results/'consistency',config)
    folder = results/'ablation'
    if (folder/'metadata.json').exists():
        model_config = json.loads((folder/'metadata.json').read_text())['model_config']
        signature = stage_signature('ablation',config,paths,{'model_config':model_config,
            'evaluation':behavioral_records(records,'validation'),'discovery':discovery,'shortlist':shortlist.to_dict('records')})
        if cached_stage(folder,signature):
            logging.info('Completed ablation cache verified; skipping model load')
            return
    model,tokenizer,_ = load_model(config)
    validate_model_inputs(model,tokenizer,[*discovery,*records],results/'attention/metadata.json')
    output = run_ablation(model,records,discovery,shortlist,folder,config,paths)
    logging.info('Ablation summary: %s',output)


if __name__=='__main__':
    main()
