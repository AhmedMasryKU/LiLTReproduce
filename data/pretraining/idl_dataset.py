import json, os
import random
from typing import Any, List, Tuple
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer 
from datasets import load_dataset, load_from_disk
import numpy as np

def normalize_bbox(bbox):
    return [int(x*1000) for x in bbox]

class IDLDataset(Dataset): 

    def __init__(
        self,
        dataset_name_or_path: str,
        tokenizer : AutoTokenizer = None,
        mlm_prob=0.15,
    ):
        super().__init__()

        self.tokenizer = tokenizer 
        # Load dataset
        # List of .npy file paths (each containing 500K files, so in total we have 2M documents)
        npy_files = os.listdir(dataset_name_or_path)
        # Load all .npy files and concatenate them
        # data = []
        # for npy_file in npy_files:
        #     npy_file_path = os.path.join(dataset_name_or_path, npy_file)
        #     data.extend(np.load(npy_file_path, allow_pickle=True)[1:])

        self.data_path = dataset_name_or_path
        self.dataset = npy_files #data
        self.dataset_length = len(self.dataset)
        self.mlm_prob = mlm_prob

    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        #row = self.dataset[idx]
        with open(os.path.join(self.data_path, self.dataset[idx])) as f:
            row = json.load(f)
        words = row["ocr_tokens"]
        bboxes = row["ocr_normalized_boxes"]
        normalized_bboxes = [normalize_bbox(bbox) for bbox in bboxes]
        # Encode inputs
        encoding = self.tokenizer(words, 
                                boxes=normalized_bboxes, 
                                truncation=True, 
                                padding="max_length", 
                                return_special_tokens_mask=True,
                                 return_tensors="pt")

        input_ids = encoding.input_ids.numpy()
        # Create MLM Labels
        labels = input_ids.copy()
        tokens_probs_matrix = np.full(labels.shape, self.mlm_prob)
        # Remove Special tokens such as padding from consideration. 
        special_tokens_mask = encoding.special_tokens_mask.numpy().astype("bool")
        tokens_probs_matrix[special_tokens_mask] = 0.0

        masked_tokens_indices = np.random.binomial(1, tokens_probs_matrix).astype("bool")
        # Avoid computing loss for other Tokens shouldn't 
        labels[~masked_tokens_indices] = -100

        # <mask> token for 80% of the masked tokens. 
        tokens_replaced_with_mask = np.random.binomial(1, np.full(labels.shape, 0.8)).astype("bool") & masked_tokens_indices
        input_ids[tokens_replaced_with_mask] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # Random Token for 10% of the masked tokens. 
        random_tokens_indices = np.random.binomial(1, np.full(labels.shape, 0.5)).astype("bool")
        random_tokens_indices &= masked_tokens_indices & ~tokens_replaced_with_mask
        selected_random_tokens = np.random.randint(self.tokenizer.vocab_size, size=labels.shape, dtype="i4")
        input_ids[random_tokens_indices] = selected_random_tokens[random_tokens_indices]

        return {"input_ids": torch.LongTensor(input_ids).squeeze(0), 
                "bbox": encoding.bbox.squeeze(0),
                "attention_mask": encoding.attention_mask.squeeze(0), 
                "labels": torch.LongTensor(labels).squeeze(0)
        }
    


class IDLDatasetNpy(Dataset): 

    def __init__(
        self,
        dataset_name_or_path: str,
        tokenizer : AutoTokenizer = None,
        mlm_prob=0.15,
    ):
        super().__init__()

        self.tokenizer = tokenizer 
        # Load dataset
        # List of .npy file paths (each containing 500K files, so in total we have 2M documents)
        npy_files = os.listdir(dataset_name_or_path)
        # Load all .npy files and concatenate them
        data = []
        for npy_file in npy_files:
            npy_file_path = os.path.join(dataset_name_or_path, npy_file)
            data.extend(np.load(npy_file_path, allow_pickle=True)[1:])

        self.data_path = dataset_name_or_path
        self.dataset = data
        self.dataset_length = len(self.dataset)
        self.mlm_prob = mlm_prob

    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        row = self.dataset[idx]

        words = row["ocr_tokens"]
        bboxes = row["ocr_normalized_boxes"]
        normalized_bboxes = [normalize_bbox(bbox) for bbox in bboxes]
        # Encode inputs
        encoding = self.tokenizer(words, 
                                boxes=normalized_bboxes, 
                                truncation=True, 
                                padding="max_length", 
                                return_special_tokens_mask=True,
                                 return_tensors="pt")

        input_ids = encoding.input_ids.numpy()
        # Create MLM Labels
        labels = input_ids.copy()
        tokens_probs_matrix = np.full(labels.shape, self.mlm_prob)
        # Remove Special tokens such as padding from consideration. 
        special_tokens_mask = encoding.special_tokens_mask.numpy().astype("bool")
        tokens_probs_matrix[special_tokens_mask] = 0.0

        masked_tokens_indices = np.random.binomial(1, tokens_probs_matrix).astype("bool")
        # Avoid computing loss for other Tokens shouldn't 
        labels[~masked_tokens_indices] = -100

        # <mask> token for 80% of the masked tokens. 
        tokens_replaced_with_mask = np.random.binomial(1, np.full(labels.shape, 0.8)).astype("bool") & masked_tokens_indices
        input_ids[tokens_replaced_with_mask] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # Random Token for 10% of the masked tokens. 
        random_tokens_indices = np.random.binomial(1, np.full(labels.shape, 0.5)).astype("bool")
        random_tokens_indices &= masked_tokens_indices & ~tokens_replaced_with_mask
        selected_random_tokens = np.random.randint(self.tokenizer.vocab_size, size=labels.shape, dtype="i4")
        input_ids[random_tokens_indices] = selected_random_tokens[random_tokens_indices]

        return {"input_ids": torch.LongTensor(input_ids).squeeze(0), 
                "bbox": encoding.bbox.squeeze(0),
                "attention_mask": encoding.attention_mask.squeeze(0), 
                "labels": torch.LongTensor(labels).squeeze(0)
        }