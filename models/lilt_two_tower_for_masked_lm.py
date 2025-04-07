from .lilt_two_tower_model import LiltTwoTowerModel
from .lilt_for_masked_lm import LiltLMHead
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel
from typing import Optional, Union, Tuple
import torch 
import torch.nn as nn
from transformers.modeling_outputs import MaskedLMOutput

class LiltTwoTowerForMaskedLM(RobertaPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)

        self.lilt = LiltTwoTowerModel(config, add_pooling_layer=False)
        self.lm_head = LiltLMHead(config)

        # Initialize weights
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.lm_head = new_embeddings

    def get_input_embeddings(self):
        return self.lilt.embeddings.word_embeddings
    
    def set_input_embeddings(self, value):
        self.lilt.embeddings.word_embeddings = value

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
