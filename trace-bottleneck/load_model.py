#!/usr/bin/env python3
"""
Script to load a saved model for inference and reproducibility.
"""

import json
import pickle
from pathlib import Path
from omegaconf import OmegaConf
import pytorch_lightning as pl
import torch


def load_saved_model(model_name, saved_models_dir=None):
    """
    Load a saved model with its configuration and metadata.
    
    Args:
        model_name: Name of the saved model directory
        saved_models_dir: Path to saved_models directory (default: ./saved_models)
    
    Returns:
        tuple: (model, config, metadata)
            - model: Loaded PyTorch model
            - config: Hydra configuration OmegaConf object
            - metadata: Model metadata dictionary
    """
    if saved_models_dir is None:
        saved_models_dir = Path(__file__).parent / "saved_models"
    else:
        saved_models_dir = Path(saved_models_dir)
    
    model_path = saved_models_dir / model_name
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    # Load metadata
    metadata_path = model_path / "model_metadata.json"
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    print(f"✓ Loaded metadata: {metadata}")
    
    # Load configuration
    config_path = model_path / ".hydra" / "config.yaml"
    if config_path.exists():
        config = OmegaConf.load(config_path)
        print(f"✓ Loaded configuration from {config_path}")
    else:
        config = None
        print("⚠ Configuration file not found")
    
    # Find and load the best checkpoint
    checkpoints_dir = model_path / "checkpoints"
    if checkpoints_dir.exists():
        # Prefer 'last.ckpt' for reproducibility
        ckpt_path = checkpoints_dir / "last.ckpt"
        if not ckpt_path.exists():
            # Fall back to any .ckpt file
            ckpt_files = list(checkpoints_dir.glob("*.ckpt"))
            if ckpt_files:
                ckpt_path = ckpt_files[0]
            else:
                ckpt_path = None
        
        if ckpt_path:
            print(f"✓ Found checkpoint: {ckpt_path}")
            # Note: Loading the model requires the engine class definition
            # For now, we just return the checkpoint path
            model = ckpt_path
        else:
            model = None
            print("⚠ No checkpoints found")
    else:
        model = None
        print("⚠ Checkpoints directory not found")
    
    # Load graph and policy if they exist
    graph_path = model_path / "graph.pkl"
    policy_path = model_path / "policy.pkl"
    
    if graph_path.exists():
        with open(graph_path, 'rb') as f:
            metadata['graph'] = pickle.load(f)
        print(f"✓ Loaded graph")
    
    if policy_path.exists():
        with open(policy_path, 'rb') as f:
            metadata['policy'] = pickle.load(f)
        print(f"✓ Loaded intervention policy")
    
    return model, config, metadata


def load_model_with_engine(model_name, engine_class, saved_models_dir=None, device="auto"):
    """
    Load a saved model and initialize the engine with it.
    
    Args:
        model_name: Name of the saved model directory
        engine_class: The engine class to instantiate
        saved_models_dir: Path to saved_models directory
        device: Device to load onto ('cpu', 'cuda', or 'auto')
    
    Returns:
        tuple: (engine, config, metadata)
            - engine: Initialized engine with loaded model
            - config: Hydra configuration
            - metadata: Model metadata
    """
    ckpt_path, config, metadata = load_saved_model(model_name, saved_models_dir)
    
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load checkpoint into engine
    if ckpt_path and isinstance(ckpt_path, Path):
        engine = engine_class.load_from_checkpoint(str(ckpt_path))
        engine = engine.to(device)
        print(f"✓ Loaded engine from checkpoint on {device}")
    else:
        raise ValueError(f"Invalid checkpoint path: {ckpt_path}")
    
    return engine, config, metadata


def list_saved_models(saved_models_dir=None):
    """List all available saved models."""
    if saved_models_dir is None:
        saved_models_dir = Path(__file__).parent / "saved_models"
    else:
        saved_models_dir = Path(saved_models_dir)
    
    if not saved_models_dir.exists():
        print("No saved models directory found")
        return []
    
    models = []
    for model_dir in saved_models_dir.iterdir():
        if model_dir.is_dir():
            metadata_path = model_dir / "model_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    models.append({
                        'name': model_dir.name,
                        'created_at': metadata.get('created_at'),
                        'macro_f1': metadata.get('metrics', {}).get('macro_f1'),
                    })
    
    return models


if __name__ == "__main__":
    # Example: list available models
    print("Available saved models:")
    models = list_saved_models()
    for model in models:
        print(f"  - {model['name']}: F1={model['macro_f1']} ({model['created_at']})")
