from transformers import AutoModel, AutoTokenizer, AutoConfig
from models.lilt_two_tower_for_masked_lm import LiltTwoTowerForMaskedLM
import os 

infoxlm_model = AutoModel.from_pretrained("microsoft/infoxlm-base")
tokenizer = AutoTokenizer.from_pretrained("microsoft/layoutxlm-base")

# Two Tower Model
model = LiltTwoTowerForMaskedLM.from_pretrained("/home/masry20/projects/def-enamul/masry20/lilt_tmp_checkpoints/try_2_mlm_two_tower/checkpoint-58595")

final_model = LiltTwoTowerForMaskedLM(config=AutoConfig.from_pretrained('configs/two_tower_multilingual_config'))
# Load weights
final_model.lilt.load_state_dict(infoxlm_model.state_dict(), strict=False)

final_model.lilt.layout_encoder.load_state_dict(model.lilt.layout_encoder.state_dict())
final_model.lilt.layout_embeddings.load_state_dict(model.lilt.layout_embeddings.state_dict())

final_model.push_to_hub("ahmed-masry/lilt-two-tower-58595-multilingual", token=os.environ['HF_TOKEN'], safe_serialization=False)
tokenizer.push_to_hub("ahmed-masry/lilt-two-tower-58595-multilingual", token=os.environ['HF_TOKEN'])