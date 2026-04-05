import os 
import mygene
import pandas as pd
from pathlib import Path
import pickle

def pickle_data(data, path):
    with open(path, "wb") as f: 
        pickle.dump(data, f)
        
def load_pickle(path):
    with open(path, "rb") as f: 
        return pickle.load(f)

def process_AURORA_genes(AURORA_gene_sets_df, selected_genes):

    mg = mygene.MyGeneInfo()

    signatures = {}

    for v in AURORA_gene_sets_df.values: 

        name = v[0]
        entrez_id = [str(r) for r in v[4:] if r != "-"]
        gene_symbol_info = mg.getgenes(entrez_id, fields="symbol")
        gene_symbol = [r["symbol"] for r in gene_symbol_info if "symbol" in r]
        gene_symbol = [g for g in gene_symbol if g in selected_genes]

        if len(gene_symbol) == 0: continue 

        signatures[name] = gene_symbol
    
    return signatures

def process_immuno_genes(immuno_gene_sets_df, selected_genes):

    signatures = {}

    for v in immuno_gene_sets_df.values:

        name = v[3]
        gene_symbol = [g.strip() for g in v[4].split(":")]
        gene_symbol = [g for g in gene_symbol if g in selected_genes]

        signatures[name] = gene_symbol

    return signatures

if __name__ == "__main__":

    raw_data_folder = Path("../data/gene_sets/raw/")
    processed_data_folder = Path("../data/gene_sets/processed/")

    genes_path = "../data/TCGA/shared_genes_ordered_by_TCGA_variance.pkl"
    AURORA_gene_sets_path = raw_data_folder / "AURORA_RNA_expression_signature_gene_annotations.xlsx"
    immuno_gene_sets_path = raw_data_folder / "Immuno_RNA_expression_signature_gene_annotations.xlsx"

    # Get the gene symbols in the gene sets
    selected_genes = list(load_pickle(genes_path).keys())
    AURORA_gene_sets_df = pd.read_excel(AURORA_gene_sets_path).fillna("-")
    immuno_gene_sets_df = pd.read_excel(immuno_gene_sets_path)

    # Write the genes sets
    AURORA_output_path = processed_data_folder / "AURORA_gene_sets.pkl"
    if not os.path.exists(AURORA_output_path):
        print("Getting the genes sets for AURORA")
        AURORA_gene_sets = process_AURORA_genes(AURORA_gene_sets_df, selected_genes)
        pickle_data(AURORA_gene_sets, AURORA_output_path)

    immuno_output_path = processed_data_folder / "immuno_gene_sets.pkl"
    if not os.path.exists(immuno_output_path):
        print("Getting the genes sets for the immunotherapy")
        immuno_gene_sets = process_immuno_genes(immuno_gene_sets_df, selected_genes)
        pickle_data(immuno_gene_sets, immuno_output_path)