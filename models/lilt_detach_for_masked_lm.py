from transformers import LiltPreTrainedModel, LiltModel
# from .lilt_model import LiltModel
from transformers.modeling_outputs import MaskedLMOutput
from transformers.activations import gelu 
import torch.nn as nn 
import torch 
import math


from typing import Optional, Union, Tuple

# Monkey Patching for Applying DETACH Operation. 
from transformers.models.lilt.modeling_lilt import LiltSelfAttention

def new_forward(
        self,
        hidden_states,
        layout_inputs,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
    ):
        layout_value_layer = self.transpose_for_scores(self.layout_value(layout_inputs), r=self.channel_shrink_ratio)
        layout_key_layer = self.transpose_for_scores(self.layout_key(layout_inputs), r=self.channel_shrink_ratio)
        layout_query_layer = self.transpose_for_scores(self.layout_query(layout_inputs), r=self.channel_shrink_ratio)

        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        layout_attention_scores = torch.matmul(layout_query_layer, layout_key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        tmp_attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        tmp_layout_attention_scores = layout_attention_scores/ math.sqrt(
            self.attention_head_size // self.channel_shrink_ratio
        )
        # BiACM Module. 
        # print("tmp_attention_scores", tmp_attention_scores)
        # print("tmp_layout_attention_scores", tmp_layout_attention_scores)
        attention_scores = tmp_attention_scores + tmp_layout_attention_scores
        layout_attention_scores = tmp_layout_attention_scores + tmp_attention_scores.detach()

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            layout_attention_scores = layout_attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        layout_attention_probs = nn.Softmax(dim=-1)(layout_attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        layout_attention_probs = self.dropout(layout_attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            layout_attention_probs = layout_attention_probs * head_mask

        layout_context_layer = torch.matmul(layout_attention_probs, layout_value_layer)

        layout_context_layer = layout_context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = layout_context_layer.size()[:-2] + (self.all_head_size // self.channel_shrink_ratio,)
        layout_context_layer = layout_context_layer.view(*new_context_layer_shape)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in RobertaModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (
            ((context_layer, layout_context_layer), attention_probs)
            if output_attentions
            else ((context_layer, layout_context_layer),)
        )

        return outputs

# Replace the forward with the new forward
LiltSelfAttention.forward = new_forward


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

class LiltDetachForMaskedLM(LiltPreTrainedModel):

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
            return_dict=True,
        )

        sequence_output = outputs.last_hidden_state
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
