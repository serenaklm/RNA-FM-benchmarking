"""
Convert a Python dictionary of gene sets to GMT format.

GMT format: each line is a gene set
  <set_name>\t<description>\t<gene1>\t<gene2>\t<gene3>\t...

Example dictionary:
  gene_sets = {
      'SET_NAME_1': ['GENE1', 'GENE2', 'GENE3'],
      'SET_NAME_2': ['GENE4', 'GENE5', 'GENE6']
  }
"""

import os
import pickle

def dict_to_gmt(gene_sets_dict, output_path, description="na"):
    """
    Convert dictionary of gene sets to GMT file.
    
    Parameters:
    -----------
    gene_sets_dict : dict
    output_path : str
    description : str
    """
    with open(output_path, 'w') as f:
        for set_name, genes in gene_sets_dict.items():
            # Ensure genes are strings and remove any empty strings
            genes = [str(g).strip() for g in genes if g]
            
            # Write: set_name\tdescription\tgene1\tgene2\t...
            line = f"{set_name}\t{description}\t" + "\t".join(genes) + "\n"
            f.write(line)
    
    print(f"GMT file saved to: {output_path}")
    print(f"Number of gene sets: {len(gene_sets_dict)}")
    print(f"Total genes: {sum(len(genes) for genes in gene_sets_dict.values())}")

if __name__ == "__main__":
    
    #gene_set_files = ['18_hallmark_gene_sets.pkl', '50_hallmark_gene_sets.pkl', 'immuno_gene_sets.pkl', 'AURORA_gene_sets.pkl', 'combined_gene_sets.pkl']
    #descriptions = ['18 Hallmark gene sets', '50 Hallmark gene sets', 'Immuno gene sets', 'AURORA gene sets', 'Combined hallmark, immuno, and AURORA gene sets']
    gene_set_files = ['50_hallmark_gene_sets.pkl']
    descriptions = ['50 Hallmark gene sets']

    for f, desc in zip(gene_set_files, descriptions):
        path = os.path.join('../data/gene_sets/processed/', f)
        with open(path, 'rb') as file:
            gene_sets = pickle.load(file)
        output_gmt = f.replace('.pkl', '.gmt')
        dict_to_gmt(gene_sets, os.path.join('../data/gene_sets/processed', output_gmt), desc)