import json
import yaml 
import h5py
import pickle 

def load_config(path):
    with open(path, "r") as f: 
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def _patch_numpy_core():
    """Shim numpy._core → numpy.core for pickles saved with NumPy 2.x loaded in NumPy 1.x."""
    import sys
    import types
    import numpy as np
    if hasattr(np, "_core"):
        return  # already present (numpy 2.x), nothing to do
    # Create a minimal numpy._core module that maps to numpy.core
    core = sys.modules.get("numpy.core")
    shim = types.ModuleType("numpy._core")
    for attr in dir(core):
        try:
            setattr(shim, attr, getattr(core, attr))
        except Exception:
            pass
    sys.modules["numpy._core"] = shim
    for submod in ["multiarray", "numeric", "umath", "_multiarray_umath", "fromnumeric"]:
        full_old = f"numpy.core.{submod}"
        full_new = f"numpy._core.{submod}"
        if full_old in sys.modules and full_new not in sys.modules:
            sys.modules[full_new] = sys.modules[full_old]


def load_pickle(path):
    with open(path, "rb") as f:
        try:
            return pickle.load(f)
        except ModuleNotFoundError as e:
            if "numpy._core" not in str(e):
                raise
            # Pickle was saved with NumPy 2.x; apply shim and retry
            _patch_numpy_core()
    with open(path, "rb") as f:
        return pickle.load(f)

def load_json(path):
    with open(path, "r", encoding = "UTF-8") as f: 
        return json.load(f)

def load_h5(path):
    
    with h5py.File(path, "r") as f:
        feats = f["features"][:]
        return feats

def pickle_data(data, path):
    with open(path, "wb") as f: 
        pickle.dump(data, f)

def write_json(data, path):
    with open(path, "w", encoding = "UTF-8") as f: 
        json.dump(data, f, indent = 4)

def checkpoint_every_n_train_steps(config, dataset_name):
    """Checkpoint cadence for ModelCheckpoint.

    Default (None): the checkpoint is considered at the end of every validation epoch, so the
    saved model is the one from the epoch with the lowest validation loss.
    Datasets listed in config["callbacks"]["legacy_step_checkpoint_datasets"] instead check every
    trainer.log_every_n_steps optimizer steps, which is how their reported models were trained
    (the saved model can then be from shortly after the best validation epoch).
    """
    legacy = config.get("callbacks", {}).get("legacy_step_checkpoint_datasets") or []
    return config["trainer"]["log_every_n_steps"] if dataset_name in legacy else None
