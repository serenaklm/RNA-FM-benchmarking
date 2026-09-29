import os
import json
import requests
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

outdir = os.environ.get("TCGA_COUNTS_OUTPUT_DIR", "/data/rbg/users/ashyamal/data/new_TCGA_RNAseq_STAR_counts/")
os.makedirs(outdir, exist_ok=True)

# TCGA project list (33 projects)
tcga_projects = [
    'TCGA-BRCA', 'TCGA-GBM','TCGA-OV','TCGA-LUAD', 'TCGA-UCEC', 'TCGA-KIRC',
    'TCGA-HNSC','TCGA-LGG','TCGA-THCA','TCGA-LUSC','TCGA-PRAD','TCGA-SKCM',
    'TCGA-COAD','TCGA-STAD','TCGA-BLCA','TCGA-LIHC','TCGA-CESC','TCGA-KIRP',
    'TCGA-SARC','TCGA-LAML','TCGA-ESCA','TCGA-PAAD','TCGA-PCPG','TCGA-READ',
    'TCGA-TGCT','TCGA-THYM','TCGA-KICH','TCGA-ACC','TCGA-MESO','TCGA-UVM',
    'TCGA-DLBC','TCGA-UCS','TCGA-CHOL'
]

BASE_API = "https://api.gdc.cancer.gov/"

# -----------------------------
# Helper functions
# -----------------------------
def query_gdc(project):
    """Query GDC API for STAR counts primary tumor files for a project"""
    filters = {
        "op": "and",
        "content": [
            {"op": "in", "content": {"field": "cases.project.project_id", "value": [project]}},
            {"op": "in", "content": {"field": "files.data_category", "value": ["Transcriptome Profiling"]}},
            {"op": "in", "content": {"field": "files.data_type", "value": ["Gene Expression Quantification"]}},
            {"op": "in", "content": {"field": "files.experimental_strategy", "value": ["RNA-Seq"]}},
            {"op": "in", "content": {"field": "files.analysis.workflow_type", "value": ["STAR - Counts"]}},
            #{"op": "in", "content": {"field": "cases.samples.sample_type", "value": ["Primary Tumor", "Primary Blood Derived Cancer - Peripheral Blood"]}} # Include only primary tumor samples + "Primary Blood Derived Cancer - Peripheral Blood" for TCGA-LAML
        ]
    }
    params = {
        "filters": json.dumps(filters),
        "format": "JSON",
        "fields": "file_id,file_name,cases.case_id,cases.samples.sample_type",
        "size": "10000"
    }
    r = requests.get(BASE_API + "files", params=params)
    r.raise_for_status()
    files = r.json()["data"]["hits"]
    return files

def download_file(file_id, outpath):
    """Download file from GDC with streaming"""
    url = f"https://api.gdc.cancer.gov/data/{file_id}"
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    with open(outpath, "wb") as fh:
        for chunk in resp.iter_content(chunk_size=1024*1024):
            if chunk:
                fh.write(chunk)

def process_project(proj):
    """Process a single TCGA project"""
    try:
        print(f"\n=== Processing {proj} ===")
        project_dir = os.path.join(outdir, proj)
        
        # Check if project already completed
        counts_matrix_file = os.path.join(project_dir, "rnaSeq_counts_matrix.csv")
        if os.path.exists(counts_matrix_file):
            print(f"{proj} already downloaded (counts matrix exists). Skipping...")
            return f"{proj}: Already completed"
        
        os.makedirs(project_dir, exist_ok=True)
        
        files = query_gdc(proj)
        
        if len(files) == 0:
            print(f"No STAR counts found for {proj}")
            return f"{proj}: No files found"
        
        print(f"Found {len(files)} files for {proj}")
        
        dfs = []
        sample_metadata = []
        
        # Progress bar for processing files
        for f in tqdm(files, desc=f"Processing {proj} files", unit="file", position=0, leave=True):
            file_id = f['file_id']
            file_name = f['file_name']
            case_id = f['cases'][0]['case_id']
            
            filename = os.path.join(project_dir, f"{case_id}_{file_name}")
            
            if not os.path.exists(filename):
                download_file(file_id, filename)
            
            # Read counts - STAR counts files have gene_id, gene_name, gene_type, and count columns
            df = pd.read_csv(filename, sep="\t", comment='#', skiprows=1)
            
            # Use the unstranded column (column index 3) or stranded_first/second depending on your needs
            # Columns are typically: gene_id, gene_name, gene_type, unstranded, stranded_first, stranded_second
            if 'unstranded' in df.columns:
                counts_col = 'unstranded'
            else:
                # Fallback to 4th column if column names differ
                counts_col = df.columns[3]
            
            count_df = pd.DataFrame({
                case_id: df[counts_col].values
            }, index=df['gene_id'])
            
            dfs.append(count_df)
            
            sample_metadata.append({
                "sample_id": case_id,
                "file_id": file_id,
                "file_name": file_name
            })
        
        # Merge all counts into a single matrix
        counts_matrix = pd.concat(dfs, axis=1)
        counts_matrix.to_csv(os.path.join(project_dir, "rnaSeq_counts_matrix.csv"))
        
        # Save sample metadata
        sample_df = pd.DataFrame(sample_metadata)
        sample_df.to_csv(os.path.join(project_dir, "rnaSeq_sample_table.csv"), index=False)
        
        # Print summary stats
        summary = f"{proj} summary: {counts_matrix.shape[1]} samples, {counts_matrix.shape[0]} genes"
        print(summary)
        
        return f"{proj}: Completed successfully ({counts_matrix.shape[1]} samples)"
        
    except Exception as e:
        error_msg = f"{proj}: ERROR - {str(e)}"
        print(error_msg)
        return error_msg

if __name__ == '__main__':
    num_workers = 6
    
    print(f"Starting parallel download with {num_workers} workers...")
    print(f"Total projects to process: {len(tcga_projects)}\n")
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all projects
        future_to_project = {executor.submit(process_project, proj): proj for proj in tcga_projects}
        
        completed = 0
        total = len(tcga_projects)
        
        for future in as_completed(future_to_project):
            proj = future_to_project[future]
            completed += 1
            try:
                result = future.result()
                print(f"\n[{completed}/{total}] {result}")
            except Exception as exc:
                print(f"\n[{completed}/{total}] {proj} generated an exception: {exc}")
    
    print("\n All downloads completed")