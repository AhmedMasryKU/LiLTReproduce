from transformers import AutoModel

model1 = AutoModel.from_pretrained("/home/masry20/projects/def-enamul/masry20/lilt_tmp_checkpoints/try_2_mlm/checkpoint-46876")
model2 = AutoModel.from_pretrained("/home/masry20/projects/def-enamul/masry20/lilt_tmp_checkpoints/try_2_mlm_detach/checkpoint-46876")

# model.push_to_hub("ahmed-masry/lilt-mlm-58595", token=os.environ['HF_TOKEN'])