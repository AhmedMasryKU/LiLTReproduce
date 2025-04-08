from transformers import AutoModel, AutoTokenizer
from models.lilt_two_tower_for_masked_lm import LiltTwoTowerForMaskedLM
import os 

# model1 = AutoModel.from_pretrained("/home/masry20/projects/def-enamul/masry20/lilt_tmp_checkpoints/try_2_mlm/checkpoint-46876")
# model2 = AutoModel.from_pretrained("/home/masry20/projects/def-enamul/masry20/lilt_tmp_checkpoints/try_2_mlm_detach/checkpoint-46876")

# Two Tower Model
model3 = LiltTwoTowerForMaskedLM.from_pretrained("/home/masry20/projects/def-enamul/masry20/lilt_tmp_checkpoints/try_2_mlm_two_tower/checkpoint-58595")
tokenizer3 = AutoTokenizer.from_pretrained('ahmed-masry/lilt-two-tower-base')
#model.push_to_hub("ahmed-masry/lilt-mlm-58595", token=os.environ['HF_TOKEN'])

model3.push_to_hub("ahmed-masry/lilt-two-tower-58595", token=os.environ['HF_TOKEN'], safe_serialization=False)
tokenizer3.push_to_hub("ahmed-masry/lilt-two-tower-58595", token=os.environ['HF_TOKEN'])