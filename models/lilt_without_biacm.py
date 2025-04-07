from transformers import LiltPreTrainedModel, LiltModel
from transformers.modeling_outputs import MaskedLMOutput
from transformers.activations import gelu 
import torch.nn as nn 
import torch 
import math
from typing import Optional, Union, Tuple
from transformers import RobertaEncoder, RobertaEmbeddings, PreTrainedModel, RobertaPooler


class LiltModel(PreTrainedModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = RobertaEmbeddings(config)
        self.layout_embeddings = RobertaEmbeddings(config)

        self.text_encoder = RobertaEncoder(config)
        self.layout_encoder = RobertaEncoder(config)

        self.text_pooler = RobertaPooler(config)
        
