import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import requests
import json
import time
import numpy as np
from tqdm import tqdm
import sys
from collections import Counter
from itertools import chain
import requests
import re
import pickle
import glob
import anndata as ad
from pathlib import Path
import torch

def load_pickle(path): 

    with open(path, "rb") as f: 
        return pickle.load(f)
    
FM_embed_folder = Path("../../data/TCGA/RNA/FM_embeddings")

for f in os.listdir(FM_embed_folder):
    current_folder = FM_embed_folder / f
    for file in os.listdir(current_folder):
        current_path = current_folder / file
        embedding = load_pickle(current_path)
        
        # Convert to numpy based on type
        if isinstance(embedding, torch.Tensor):
            output = embedding.detach().cpu().numpy()
        elif isinstance(embedding, np.ndarray):
            output = embedding
        else:
            output = np.array(embedding)
        
        print(f, output.shape)
        break