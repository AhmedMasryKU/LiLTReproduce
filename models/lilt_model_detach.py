from transformers.models.roberta.modeling_roberta import RobertaModel, RobertaEmbeddings, RobertaPreTrainedModel, RobertaEncoder, RobertaPooler, RobertaSelfOutput, RobertaIntermediate, RobertaIntermediate, RobertaOutput
from transformers.modeling_outputs import BaseModelOutputWithPooling, BaseModelOutput, BaseModelOutputWithPastAndCrossAttentions, BaseModelOutputWithPastAndCrossAttentions, BaseModelOutputWithPoolingAndCrossAttentions
import torch
import torch.nn as nn
from typing import Optional, Tuple, Union, List
from transformers.pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
import copy, math
from transformers.models.roberta.modeling_roberta import create_position_ids_from_input_ids

# Monkey Patching Text Embeddings funciton to return position ids
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

from transformers.models.roberta.modeling_roberta import RobertaEmbeddings
RobertaEmbeddings.forward = new_text_embeddings_forward



class LiltLayoutEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Determine per-dimension size from total hidden size
        dim_per_coord = config.hidden_size // 6
        layout_dim = config.hidden_size // config.channel_shrink_ratio

        # Coordinate embeddings: x/y for left/right & top/bottom, plus height and width
        self.coord_embed = nn.ModuleDict({
            "x_left": nn.Embedding(config.max_2d_position_embeddings, dim_per_coord),
            "y_top": nn.Embedding(config.max_2d_position_embeddings, dim_per_coord),
            "height": nn.Embedding(config.max_2d_position_embeddings, dim_per_coord),
            "width": nn.Embedding(config.max_2d_position_embeddings, dim_per_coord),
        })

        # Positional encoding for sequence ids
        self.seq_position = nn.Embedding(
            num_embeddings=config.max_position_embeddings,
            embedding_dim=layout_dim,
            padding_idx=config.pad_token_id
        )

        # Transform combined spatial representation to final space
        self.spatial_proj = nn.Linear(config.hidden_size, layout_dim)
        self.norm = nn.LayerNorm(layout_dim, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    def forward(self, bbox, position_ids):
        x_l = self.coord_embed["x_left"](bbox[:, :, 0])
        y_t = self.coord_embed["y_top"](bbox[:, :, 1])
        x_r = self.coord_embed["x_left"](bbox[:, :, 2])
        y_b = self.coord_embed["y_top"](bbox[:, :, 3])

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

class LiLTSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # Layout query, key and value.
        layout_hidden_size = config.hidden_size // config.channel_shrink_ratio
        layout_all_head_size = self.all_head_size // config.channel_shrink_ratio
        self.layout_query = nn.Linear(layout_hidden_size, layout_all_head_size)
        self.layout_key = nn.Linear(layout_hidden_size, layout_all_head_size)
        self.layout_value = nn.Linear(layout_hidden_size, layout_all_head_size)


        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.config = config

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def layout_transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size // self.config.channel_shrink_ratio)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        layout_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:



        mixed_query_layer = self.query(hidden_states)
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        layout_key_layer = self.layout_transpose_for_scores(self.layout_key(layout_hidden_states))
        layout_value_layer = self.layout_transpose_for_scores(self.layout_value(layout_hidden_states))
        layout_query_layer = self.layout_transpose_for_scores(self.layout_query(layout_hidden_states))


        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        layout_attention_scores = torch.matmul(layout_query_layer, layout_key_layer.transpose(-1, -2))


        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]
  
            position_ids_l = torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
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

        # BiACM Module Here!
        attention_scores_before_biacm = attention_scores / math.sqrt(self.attention_head_size)
        layout_attention_scores_before_biacm = layout_attention_scores / math.sqrt(self.attention_head_size // self.config.channel_shrink_ratio)
        # Add Acores together!
        attention_scores = attention_scores_before_biacm + layout_attention_scores_before_biacm
        # Apply detach operation here.
        layout_attention_scores = attention_scores_before_biacm.detach() + layout_attention_scores_before_biacm

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in RobertaModel forward() function)
            attention_scores = attention_scores + attention_mask
            layout_attention_scores = layout_attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        layout_attention_probs = nn.functional.softmax(layout_attention_scores, dim=-1)

        attention_probs = self.dropout(attention_probs)
        layout_attention_probs = self.dropout(layout_attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
            layout_attention_probs = layout_attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        # layout context layers
        layout_context_layer = torch.matmul(layout_attention_probs, layout_value_layer)
        layout_context_layer = layout_context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape_layout = layout_context_layer.size()[:-2] + (self.all_head_size // self.config.channel_shrink_ratio,)
        layout_context_layer = layout_context_layer.view(new_context_layer_shape_layout)

        outputs = ((context_layer, layout_context_layer), attention_probs) if output_attentions else ((context_layer, layout_context_layer),)

        return outputs


class LiLTAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        self.self = LiLTSelfAttention(
            config, position_embedding_type=position_embedding_type
        )
        self.output = RobertaSelfOutput(config)
        # layout output
        layout_config = copy.deepcopy(config)
        layout_config.hidden_size = layout_config.hidden_size // layout_config.channel_shrink_ratio
        self.layout_output = RobertaSelfOutput(layout_config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        layout_hidden_states: torch.Tensor, # Layout Hidden States
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        self_outputs = self.self(
            hidden_states,
            layout_hidden_states,
            attention_mask,
            head_mask,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0][0], hidden_states)
        attention_output_layout = self.layout_output(self_outputs[0][1], layout_hidden_states)
        outputs = ((attention_output, attention_output_layout),) + self_outputs[1:]  # add attentions if we output them
        return outputs

class LiLTLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = LiLTAttention(config)
        self.is_decoder = False
        self.add_cross_attention = False

        self.intermediate = RobertaIntermediate(config)
        self.output = RobertaOutput(config)

        # Intermediate and Output layers for the layout stream.
        # Change the config first
        layout_config = copy.deepcopy(config)
        layout_config.hidden_size = layout_config.hidden_size // layout_config.channel_shrink_ratio
        layout_config.intermediate_size = layout_config.intermediate_size // layout_config.channel_shrink_ratio

        self.layout_intermediate = RobertaIntermediate(layout_config)
        self.layout_output = RobertaOutput(layout_config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        layout_hidden_states: Optional[torch.Tensor] = None, # Added layout hidden states.
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        
        self_attention_outputs = self.attention(
            hidden_states,
            layout_hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0][0]
        attention_output_layout = self_attention_outputs[0][1]

        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        layer_output_layout = apply_chunking_to_forward(
            self.feed_forward_chunk_layout, self.chunk_size_feed_forward, self.seq_len_dim, attention_output_layout
        )

        outputs = ((layer_output, layer_output_layout),) + outputs


        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

    def feed_forward_chunk_layout(self, attention_output):
        intermediate_output_layout = self.layout_intermediate(attention_output)
        layer_output_layout = self.layout_output(intermediate_output_layout, attention_output)
        return layer_output_layout

class LiLTEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # Changed RobertaLayer to LiLTLayer
        self.layer = nn.ModuleList([LiLTLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        layout_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:

        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None


        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    layout_hidden_states, # Added hidden states as inputs. 
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    layout_hidden_states, # Added hidden states as inputs. 
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            hidden_states = layer_outputs[0][0]
            layout_hidden_states = layer_outputs[0][1]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )
    

    # Copied from RobertaModel and then modified the relevant layers to become LiLT
class LiLTModel(RobertaPreTrainedModel):

    _no_split_modules = []

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = RobertaEmbeddings(config)
        self.layout_embeddings = LiltLayoutEmbeddings(config)

        self.encoder = LiLTEncoder(config)
        self.pooler = RobertaPooler(config) if add_pooling_layer else None

        self.attn_implementation = config._attn_implementation
        self.position_embedding_type = config.position_embedding_type

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        bbox: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        embedding_output, position_ids = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )
        # Embed the layouts
        layout_embeds_vectors = self.layout_embeddings(bbox, position_ids)

        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)

        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        encoder_outputs = self.encoder(
            embedding_output,
            layout_embeds_vectors, # Added layout embedding output as input.
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
