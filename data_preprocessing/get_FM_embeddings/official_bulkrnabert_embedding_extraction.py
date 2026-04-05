from huggingface_hub import hf_hub_download
import numpy as np
import pandas as pd
from transformers import AutoConfig, AutoModel, AutoTokenizer

# Load model and tokenizer.
config = AutoConfig.from_pretrained(
    "InstaDeepAI/BulkRNABert",
    trust_remote_code=True,
)
config.embeddings_layers_to_save = (4,) # last transformer layer

tokenizer = AutoTokenizer.from_pretrained("InstaDeepAI/BulkRNABert", trust_remote_code=True)
model = AutoModel.from_pretrained(
    "InstaDeepAI/BulkRNABert",
    config=config,
    trust_remote_code=True,
)

# Load bulk RNA-seq data and preprocess them.
csv_path = hf_hub_download(
    repo_id="InstaDeepAI/BulkRNABert",
    filename="data/tcga_sample.csv",
    repo_type="model",
)
gene_expression_array = pd.read_csv(csv_path).drop(["identifier"], axis=1).to_numpy()[:1, :]
gene_expression_array = np.log10(1 + gene_expression_array)
assert gene_expression_array.shape[1] == config.n_genes

# Tokenize
gene_expression_ids = tokenizer.batch_encode_plus(gene_expression_array, return_tensors="pt")["input_ids"]

# Compute BulkRNABert's embeddings
gene_expression_mean_embeddings = model(gene_expression_ids)["embeddings_4"].mean(axis=1)  # embeddings can be used for downstream tasks.
