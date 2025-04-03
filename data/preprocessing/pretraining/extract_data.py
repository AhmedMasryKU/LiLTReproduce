import numpy as np
import os
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

data_folder = "/home/masry20/scratch/idl_data/raw_data/selected_data/"
output_folder = os.path.join(os.environ['SLURM_TMPDIR'], "extracted_data")
npy_files = ["imdb_pretrain_p0.npy", "imdb_pretrain_p1.npy", "imdb_pretrain_p2.npy", "imdb_pretrain_p3.npy"]

os.makedirs(output_folder, exist_ok=True)

def process_row(i, row, npy_file):
    if isinstance(row, dict):
        row_dict = row.copy()
    else:
        row_dict = dict(row)

    for k, v in row_dict.items():
        if isinstance(v, np.ndarray):
            row_dict[k] = v.tolist()

    row_dict["original_filename"] = npy_file
    row_dict["index"] = i

    output_file = os.path.join(output_folder, f"{npy_file}_row_{i}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(row_dict, f, ensure_ascii=False, indent=2)

def process_file(npy_file):
    print("Loading file", npy_file)
    npy_file_path = os.path.join(data_folder, npy_file)
    rows = np.load(npy_file_path, allow_pickle=True)[1:]
    print("File loaded:", npy_file)

    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = [
            executor.submit(process_row, i, row, npy_file)
            for i, row in enumerate(rows)
        ]

        for _ in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {npy_file}"):
            pass  # just for tqdm update

for npy_file in npy_files:
    process_file(npy_file)
