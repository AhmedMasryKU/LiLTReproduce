from transformers import LiltPreTrainedModel, LiltModel
from transformers.modeling_outputs import MaskedLMOutput
from transformers.activations import gelu 
import torch.nn as nn 
import torch 
import math
from typing import Optional, Union, Tuple


class LiltLMHead(nn.Module):
    """LiLT Head for masked language modeling."""

    def __init__(self, config):
        super().__init__()

        self.mapping_layer = nn.Linear(config.hidden_size, config.hidden_size)
        self.norm_layer = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)

        # bias for output tokens. 
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.lm_head.bias = self.bias

    def forward(self, output_vectors, **kwargs):
        out = self.mapping_layer(output_vectors)
        out = gelu(out)
        out = self.norm_layer(out)
        
        out = self.lm_head(out)
        return out

    def _tie_weights(self):
        if self.lm_head.bias.device.type == "meta":
            self.lm_head.bias = self.bias
        else:
            self.bias = self.lm_head.bias

class LiltForMaskedLM(LiltPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)

        self.lilt = LiltModel(config, add_pooling_layer=False)
        self.lm_head = LiltLMHead(config)

        # Initialize weights
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.lm_head = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        bbox: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor],  MaskedLMOutput]:
 
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.lilt(
            input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        logits = self.lm_head(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
