import os 
import re

import numpy as np
import pandas as pd
from pathlib import Path

def parse_attr(attr_str, key):
    # Extract key "gene_id" or "gene_name" from GTF attribute field
    # Handles patterns like: gene_id "ENSG..."; gene_name "TP53";
    m = re.search(rf'{key}\s+"([^"]+)"', attr_str)
    return m.group(1) if m else np.nan

def merged_length(df):
    # df has columns: start, end for one gene; 1-based inclusive coords in GTF
    intervals = df[["start","end"]].sort_values(["start","end"]).to_numpy()
    merged = []
    for s,e in intervals:
        if not merged or s > merged[-1][1] + 1:
            merged.append([s,e])
        else:
            merged[-1][1] = max(merged[-1][1], e)
    # Length in bases (inclusive coordinates): end - start + 1
    return sum(e - s + 1 for s,e in merged)

def get_gene_length(gencode_path):

    gencode = pd.read_csv(gencode_path, sep = "\t", comment="#", 
                      names = ["seqname","source","feature","start","end","score","strand","frame","attribute"],
                      usecols = ["seqname","feature","start","end","attribute"])

    exons = gencode[gencode["feature"] == "exon"].copy()
    exons["gene_id"] = exons["attribute"].apply(lambda s: parse_attr(s, "gene_id"))
    exons["gene_name"] = exons["attribute"].apply(lambda s: parse_attr(s, "gene_name"))
    exons = exons.dropna(subset=["gene_id"])  # gene_id should exist

    gene_len_bp = exons.groupby("gene_id", sort=False).apply(merged_length, include_groups=False).astype(int)
    gene_len_kb = gene_len_bp / 1000.0

    gid_to_gname = (
        exons.drop_duplicates("gene_id")[["gene_id","gene_name"]]
        .set_index("gene_id")["gene_name"]
    )
    gene_length_table = (
        gene_len_kb.rename("length_kb").to_frame()
        .assign(gene_name=lambda df: df.index.map(gid_to_gname))
        .reset_index()[["gene_id","gene_name","length_kb"]]
    )

    return gene_length_table 

if __name__ == "__main__":

    raw_data_folder = Path("../data")
    gencode_path = raw_data_folder / "GENCODE" / "gencode.v45.annotation.gtf"
    gene_length_output_path = raw_data_folder / "GENCODE" / "gene_length.csv"

    if not os.path.exists(gene_length_output_path):
        gene_length = get_gene_length(gencode_path)
        gene_length.to_csv(gene_length_output_path, index = False)
    else:
        raise Exception(f"{gene_length_output_path} already exists. Remove this exception to get the length again")
    
