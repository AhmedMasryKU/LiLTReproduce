from transformers import RobertaConfig, RobertaForMaskedLM
from transformers import AutoTokenizer
from models.lilt_two_tower_for_masked_lm import LiltTwoTowerForMaskedLM
import os 

roberta_config = RobertaConfig.from_pretrained("configs/two_tower_config/")
roberta_model = RobertaForMaskedLM.from_pretrained("FacebookAI/roberta-base")
two_tower_model = LiltTwoTowerForMaskedLM(config=roberta_config)
tokenizer = AutoTokenizer.from_pretrained("microsoft/layoutlmv3-base")

# Copy weights
two_tower_model.lilt.embeddings.load_state_dict(roberta_model.roberta.embeddings.state_dict())
two_tower_model.lilt.encoder.load_state_dict(roberta_model.roberta.encoder.state_dict())
two_tower_model.lilt.layout_encoder.load_state_dict(roberta_model.roberta.encoder.state_dict())
#two_tower_model.lilt.pooler.load_state_dict(roberta_model.roberta.pooler.state_dict())

# Rename LM Head Weights
def rename_state_dict_keys(old_state_dict):
    new_state_dict = {}
    for key, value in old_state_dict.items():
        if key.startswith("dense."):
            new_key = key.replace("dense", "mapping_layer", 1)
        elif key.startswith("layer_norm."):
            new_key = key.replace("layer_norm", "norm_layer", 1)
        elif key.startswith("decoder."):
            new_key = key.replace("decoder", "lm_head", 1)
        else:
            new_key = key  # keep the key if not renamed
        new_state_dict[new_key] = value
    return new_state_dict


two_tower_model.lm_head.load_state_dict(rename_state_dict_keys(roberta_model.lm_head.state_dict()))

# Push Model & Tokenizer
two_tower_model.push_to_hub("ahmed-masry/lilt-two-tower-base", token=os.environ['HF_TOKEN'], safe_serialization=False)
tokenizer.push_to_hub("ahmed-masry/lilt-two-tower-base", token=os.environ['HF_TOKEN'])