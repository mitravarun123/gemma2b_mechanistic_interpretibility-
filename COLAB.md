# Run this project in Google Colab

The project runs prompts through Gemma to measure attention and causal ablation/patching; it does not train the model's weights. The second and third cells below create separate debug and 600-record configurations whose data/results persist in Google Drive.

## 1. Select a GPU and upload the project ZIP

In Colab, choose **Runtime → Change runtime type → GPU**. Run this cell and use the file picker to upload `gemma-head-specialization-colab.zip` from your computer:

```python
from google.colab import drive, files
from pathlib import Path
from zipfile import ZipFile

drive.mount('/content/drive')
uploaded = files.upload()  # Select gemma-head-specialization-colab.zip
zip_name = next(name for name in uploaded if name.endswith('.zip'))
with ZipFile(zip_name) as bundle:
    bundle.extractall('/content')
project = Path('/content/gemma-head-specialization')
%cd /content/gemma-head-specialization
```

## 2. Install dependencies and confirm GPU

```python
%pip install -e .
import torch
assert torch.cuda.is_available(), 'Choose a GPU runtime in Runtime settings'
print(torch.cuda.get_device_name(0))
free, total = torch.cuda.mem_get_info()
print(f'GPU memory: {free / 2**30:.1f} GiB free of {total / 2**30:.1f} GiB')
print('bfloat16 supported:', torch.cuda.is_bf16_supported())
```

Use a GPU with ample memory. The default checkpoint is estimated at 9.3 GB in float32, before attention/activation memory. This project returns attention weights and repeats inference for causal interventions, so ordinary inference hardware guidance is only a starting point. The project supports float32 and bfloat16 weights; the code below picks bfloat16 only when the GPU reports support and starts with batch size 1.

## 3. Authorize access and make persistent configs

First accept the license and obtain access to the selected model on Hugging Face. In Colab's Secrets panel (key icon), create `HF_TOKEN` and grant this notebook access. Then run:

```python
from google.colab import userdata
import os, shutil, yaml

os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')
run_root = Path('/content/drive/MyDrive/gemma-head-specialization-runs')
raw_source = project / 'data' / 'raw'

def make_config(label, per_task):
    run_dir = run_root / label
    raw_dir = run_dir / 'data' / 'raw'
    raw_dir.mkdir(parents=True, exist_ok=True)
    for source in raw_source.iterdir():
        if source.is_file():
            shutil.copy2(source, raw_dir / source.name)
    config = yaml.safe_load((project / 'configs' / 'debug.yaml').read_text())
    config['model']['name'] = 'google/gemma-2b'
    config['model']['device'] = 'cuda'
    config['model']['dtype'] = 'bfloat16' if torch.cuda.is_bf16_supported() else 'float32'
    config['batch_size'] = 1
    config['dataset']['discovery_per_task'] = per_task
    config['dataset']['validation_per_task'] = per_task
    config['data_dir'] = str(run_dir / 'data')
    config['results_dir'] = str(run_dir / 'results')
    config_path = project / 'configs' / f'colab_{label}.yaml'
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    return config_path

debug_config = make_config('debug', 20)
prompts600_config = make_config('prompts600', 75)
print(debug_config, prompts600_config)
```

The config uses `google/gemma-2b`, which is the original Gemma 2B model. If you mean **Gemma 2 2B**, change that line to `google/gemma-2-2b`; accept access for that exact model too. Both models need access approval.

## 4. Run and save results

Start with debug to verify access, memory and the complete workflow:

```python
!python scripts/run_pipeline.py --config configs/colab_debug.yaml
```

Then run the 600-record config:

```python
!python scripts/run_pipeline.py --config configs/colab_prompts600.yaml
```

The 600 count means 75 discovery + 75 validation records for each of four tasks (600 clean generated records). Behavioral corrupted twins are additional. All files are saved under `MyDrive/gemma-head-specialization-runs/`, so outputs/checkpoints survive a disconnected runtime. Restart a failed invocation from the stage listed in `results/pipeline_run.json`, for example:

```python
!python scripts/run_pipeline.py --config configs/colab_prompts600.yaml --from-stage 5
```

If Figure 1 reports no causally validated head, stage 7 records that figure as blocked and the runner exits nonzero after attempting the other figures. Empty causal results are valid. You can render Figures 2–11 explicitly with:

```python
!python scripts/run_pipeline.py --config configs/colab_prompts600.yaml --from-stage 7 --figures 2 3 4 5 6 7 8 9 10 11
```

The full default config uses 12,000 clean records and can exceed a Colab session. Colab hardware availability and limits change; time the debug run before choosing a larger dataset. A Drive-backed run persists safely but may write many small files more slowly than local runtime storage.

Figure 1 now annotates the saved probability in every finite attention cell, with text switching from white to black according to the cell's luminance. The sequential viridis palette stays readable for both low and high weights; annotated pronoun-to-name cells remain outlined. Missing cells are left blank, not imputed.
