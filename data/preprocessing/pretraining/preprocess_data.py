import numpy as np
import os
from datasets import Dataset, DatasetDict

data_folder = "/home/masry20/scratch/idl_data/raw_data/imdb_v2/"
# List of .npy file paths (each containing 500K files, so in total we have 2M documents)
npy_files = ["imdb_pretrain_p0.npy", "imdb_pretrain_p1.npy", "imdb_pretrain_p2.npy", "imdb_pretrain_p3.npy"]

# Load all .npy files and concatenate them
data = []
for npy_file in npy_files:
    npy_file_path = os.path.join(data_folder, npy_file)
    data.extend(np.load(npy_file_path, allow_pickle=True)[1:]) 

# Convert to Hugging Face dataset
hf_dataset = Dataset.from_list(data)

# Save the dataset to the output path
output_path = "/home/masry20/scratch/idl_data/hf_format_2M_documents_fixed/"
hf_dataset.save_to_disk(output_path)
