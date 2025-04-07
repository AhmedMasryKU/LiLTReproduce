from transformers import RobertaTokenizer, RobertaForMaskedLM
from transformers.models.roberta.modeling_roberta import RobertaModel, RobertaEmbeddings, RobertaPreTrainedModel, RobertaEncoder, RobertaPooler
from transformers.modeling_outputs import BaseModelOutputWithPooling, BaseModelOutput
import torch
import torch.nn as nn

class LiltLayoutEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Determine per-dimension size from total hidden size
        dim_per_coord = config.hidden_size // 6

        # Coordinate embeddings: x/y for left/right & top/bottom, plus height and width
        self.coord_embed = nn.ModuleDict({
            "x_left": nn.Embedding(config.max_2d_position_embeddings, dim_per_coord),
            "y_top": nn.Embedding(config.max_2d_position_embeddings, dim_per_coord),
            "x_right": nn.Embedding(config.max_2d_position_embeddings, dim_per_coord),
            "y_bottom": nn.Embedding(config.max_2d_position_embeddings, dim_per_coord),
            "height": nn.Embedding(config.max_2d_position_embeddings, dim_per_coord),
            "width": nn.Embedding(config.max_2d_position_embeddings, dim_per_coord),
        })

        # Positional encoding for sequence ids
        self.seq_position = nn.Embedding(
            num_embeddings=config.max_position_embeddings,
            embedding_dim=config.hidden_size,
            padding_idx=config.pad_token_id
        )

        # Final Transformations
        self.spatial_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    def forward(self, bbox, position_ids):
        x_l = self.coord_embed["x_left"](bbox[:, :, 0])
        y_t = self.coord_embed["y_top"](bbox[:, :, 1])
        x_r = self.coord_embed["x_right"](bbox[:, :, 2])
        y_b = self.coord_embed["y_bottom"](bbox[:, :, 3])

        h = bbox[:, :, 3] - bbox[:, :, 1]
        w = bbox[:, :, 2] - bbox[:, :, 0]
        h_embed = self.coord_embed["height"](h)
        w_embed = self.coord_embed["width"](w)

        # Combine spatial features
        combined_spatial = torch.cat([x_l, y_t, x_r, y_b, h_embed, w_embed], dim=-1)
        projected_spatial = self.spatial_proj(combined_spatial)

        # Add sequential positional embeddings
        seq_embed = self.seq_position(position_ids)
        final_embedding = projected_spatial + seq_embed

        return self.dropout(self.norm(final_embedding))


# Monkey Patching Roberta embeddings so that it returns the position ids. 
from transformers.models.roberta.modeling_roberta import create_position_ids_from_input_ids
def new_text_embeddings_forward(
    self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
):
    if position_ids is None:
        if input_ids is not None:
            # Create the position ids from the input token ids. Any padded tokens remain padded.
            position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length)
        else:
            position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

    if input_ids is not None:
        input_shape = input_ids.size()
    else:
        input_shape = inputs_embeds.size()[:-1]

    seq_length = input_shape[1]

    # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
    # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
    # issue #5664
    if token_type_ids is None:
        if hasattr(self, "token_type_ids"):
            buffered_token_type_ids = self.token_type_ids[:, :seq_length]
            buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
            token_type_ids = buffered_token_type_ids_expanded
        else:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

    if inputs_embeds is None:
        inputs_embeds = self.word_embeddings(input_ids)
    token_type_embeddings = self.token_type_embeddings(token_type_ids)

    embeddings = inputs_embeds + token_type_embeddings
    if self.position_embedding_type == "absolute":
        position_embeddings = self.position_embeddings(position_ids)
        embeddings += position_embeddings
    embeddings = self.LayerNorm(embeddings)
    embeddings = self.dropout(embeddings)
    return embeddings, position_ids

RobertaEmbeddings.forward = new_text_embeddings_forward


# Create two towe model by adapting the RoBERTa Model from Hugginface. 
from typing import Optional, Tuple, Union
class LiltTwoTowerModel(RobertaModel):

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        # Text Tower
        self.embeddings = RobertaEmbeddings(config)
        self.encoder = RobertaEncoder(config)

        # Layout Tower
        self.layout_embeddings = LiltLayoutEmbeddings(config)
        self.layout_encoder = RobertaEncoder(config)

        # Pooler
        self.pooler = RobertaPooler(config) if add_pooling_layer else None

        self.attn_implementation = config._attn_implementation
        self.position_embedding_type = config.position_embedding_type

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        bbox: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPooling]:

      # Get input shape. 
      input_shape = input_ids.size()
      bz, sequence_len = input_shape
      device = input_ids.device

      # Resolve token type ids similar to the original Roberta
      if token_type_ids is None:
        if hasattr(self.embeddings, "token_type_ids"):
            buffered_token_type_ids = self.embeddings.token_type_ids[:, :sequence_len]
            buffered_token_type_ids_expanded = buffered_token_type_ids.expand(bz, sequence_len)
            token_type_ids = buffered_token_type_ids_expanded
        else:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
      
      extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)
      head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)


      # Embed the text and layout. 
      text_embedding_output, position_ids = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )
      layout_embedding_output = self.layout_embeddings(bbox=bbox, position_ids=position_ids)

      # Encode the text.
      text_encoder_outputs = self.encoder(
            text_embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
      )
      # Encode the layout.
      layout_encoder_outputs = self.layout_encoder(
            layout_embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
      )

      # Sum the layout and text outputs.
      text_sequence_output = text_encoder_outputs[0]
      layout_sequence_output = layout_encoder_outputs[0]

      sequence_output = (text_sequence_output + layout_sequence_output) / 2

      # Pooling
      pooled_output = None
      if self.pooler is not None:
          pooled_output = self.pooler(sequence_output)

      return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
      )