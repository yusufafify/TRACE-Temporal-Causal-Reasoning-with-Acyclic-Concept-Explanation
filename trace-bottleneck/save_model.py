#!/usr/bin/env python3
"""
Script to save and archive trained models for reproducibility.
Copies all necessary artifacts from a specific run to a dedicated saved_models directory.
"""

import os
import shutil
import json
import pickle
from pathlib import Path
from datetime import datetime
import argparse


def save_model(run_dir, model_name=None, metric_f1=None):
    """
    Save all artifacts from a model run to the saved_models directory.
    
    Args:
        run_dir: Path to the run directory (e.g., outputs/2026-04-02/00-39-43)
        model_name: Optional name for the saved model
        metric_f1: Optional F1 score to document
    """
    # Validate run directory exists
    run_path = Path(run_dir)
    if not run_path.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    
    # Create saved_models directory
    saved_models_dir = Path(__file__).parent / "saved_models"
    saved_models_dir.mkdir(exist_ok=True)
    
    # Generate model directory name
    if model_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"model_{timestamp}"
    
    model_save_path = saved_models_dir / model_name
    model_save_path.mkdir(exist_ok=True)
    
    # Copy checkpoints
    checkpoints_src = run_path / "checkpoints"
    if checkpoints_src.exists():
        checkpoints_dest = model_save_path / "checkpoints"
        if checkpoints_dest.exists():
            shutil.rmtree(checkpoints_dest)
        shutil.copytree(checkpoints_src, checkpoints_dest)
        print(f"✓ Copied checkpoints to {checkpoints_dest}")
    
    # Copy configuration
    hydra_src = run_path / ".hydra"
    if hydra_src.exists():
        hydra_dest = model_save_path / ".hydra"
        if hydra_dest.exists():
            shutil.rmtree(hydra_dest)
        shutil.copytree(hydra_src, hydra_dest)
        print(f"✓ Copied Hydra configuration to {hydra_dest}")
    
    # Copy graph and policy
    for file in ["graph.pkl", "policy.pkl"]:
        src_file = run_path / file
        if src_file.exists():
            dest_file = model_save_path / file
            shutil.copy2(src_file, dest_file)
            print(f"✓ Copied {file}")
    
    # Copy architecture documentation
    arch_src = run_path / "architecture.txt"
    if arch_src.exists():
        arch_dest = model_save_path / "architecture.txt"
        shutil.copy2(arch_src, arch_dest)
        print(f"✓ Copied architecture.txt")
    
    # Copy results
    results_src = run_path / "results"
    if results_src.exists():
        results_dest = model_save_path / "results"
        if results_dest.exists():
            shutil.rmtree(results_dest)
        shutil.copytree(results_src, results_dest)
        print(f"✓ Copied results to {results_dest}")
    
    # Create metadata file
    metadata = {
        "model_name": model_name,
        "created_at": datetime.now().isoformat(),
        "source_run": str(run_path),
        "metrics": {
            "macro_f1": metric_f1,
        },
        "files": {
            "checkpoint_latest": "checkpoints/last.ckpt",
            "configuration": ".hydra/config.yaml",
            "graph": "graph.pkl",
            "policy": "policy.pkl",
            "architecture": "architecture.txt",
            "results": "results/",
        }
    }
    
    metadata_path = model_save_path / "model_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Created metadata file: {metadata_path}")
    
    # Create README
    readme_path = model_save_path / "README.md"
    readme_content = f"""# Saved Model: {model_name}

## Model Information
- **Created**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Source Run**: {run_path}
- **Macro F1 Score**: {metric_f1 if metric_f1 else 'N/A'}

## Directory Structure
```
{model_save_path}/
├── checkpoints/              # PyTorch Lightning checkpoints
│   ├── last.ckpt            # Last checkpoint (recommended)
│   └── epoch=*.ckpt         # Epoch-specific checkpoints
├── .hydra/                  # Hydra configuration
│   ├── config.yaml          # Full configuration used for training
│   ├── hydra.yaml          # Hydra-specific settings
│   └── overrides.yaml      # Command-line overrides
├── results/                 # Test results and metrics
│   ├── confusion_matrix.pkl
│   ├── confusion_matrix_threshold.pkl
│   ├── c_accuracy.pkl
│   ├── y_accuracy.pkl
│   └── ...
├── graph.pkl               # Learned causal graph
├── policy.pkl              # Intervention policy
├── architecture.txt        # Model architecture summary
└── model_metadata.json     # This model's metadata
```

## To Load and Use This Model

See `load_model.py` for a script to load this model and run inference.

### Quick Start
```python
from load_model import load_saved_model

model, config, metadata = load_saved_model('{model_name}')
# Use model for inference
```

## Performance Metrics
- Macro F1: {metric_f1 if metric_f1 else 'See results/'}
- See `results/` directory for detailed metrics and evaluations

## Configuration
Full training configuration available in `.hydra/config.yaml`

Key hyperparameters:
- Seed: Check config.yaml for reproducibility
- Dataset: See config.yaml for dataset parameters
- Model architecture: See architecture.txt
"""
    
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    print(f"✓ Created README.md")
    
    print(f"\n✅ Model saved successfully to: {model_save_path}")
    print(f"\nTo load this model in the future, use:")
    print(f"  from load_model import load_saved_model")
    print(f"  model, config, metadata = load_saved_model('{model_name}')")
    
    return model_save_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Save a trained model for reproducibility"
    )
    parser.add_argument(
        "run_dir",
        type=str,
        help="Path to the run directory (e.g., outputs/2026-04-02/00-39-43)"
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Name for the saved model (default: model_YYYYMMDD_HHMMSS)"
    )
    parser.add_argument(
        "--f1",
        type=float,
        default=None,
        help="Macro F1 score of the model"
    )
    
    args = parser.parse_args()
    save_model(args.run_dir, model_name=args.name, metric_f1=args.f1)
