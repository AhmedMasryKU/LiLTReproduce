# Import necessary libraries

from typing import Optional, Union, Tuple # Import type hints
import torch # PyTorch library
import torch.nn as nn # Import PyTorch neural network module for building custom deep learning layers and mdels
from transformers.modeling_outputs import TokenClassifierOutput # Import the output class for token classification
from transformers import RobertaPreTrainedModel # Import the pre-trained model class for Roberta
from .lilt_two_tower_model import LiltTwoTowerModel # Import the custom LiLT model class
from .lilt_for_token_classification import TokenClassificationHead # Import the token classification head class

# Define the LiLT model for token classification by extending the pre-trained LiLT model
class LiltTwoTowerForTokenClassification(RobertaPreTrainedModel):
    def __init__(self, config):
        """
        Initializes the LiLT model for token classification

        Args:
            config: Model configuration containing parameters for LiLT and the classifier head
        """
        super().__init__(config)
        # Initialize the base LiLT model without the pooling layer
        self.lilt = LiltTwoTowerModel(config, add_pooling_layer=False)
        # Initialize the token classification head
        self.token_classifier_head = TokenClassificationHead(config)
        # Run post-initialization steps defined in the parent class (weight initialization)
        self.post_init()

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
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        """
        Forward pass for the LiLT model used in token classification

        Args:
            input_ids: Indices of input sequence tokens in the vocabulary
            bbox: Bounding box coordinates corresponding to tokens (for visual tasks)
            attention_mask: Mask to avoid performing attention on padding tokens
            token_type_ids: Segment token indices to indicate different portions of the inputs
            position_ids: Positional indices of tokens
            head_mask: Mask to nullify selected heads of the self-attention modules
            inputs_embeds: Precomputed embeddings for the inputs
            labels: Ground truth labels for computing the token classification loss
            output_attentions: If True, include attention weights in the output
            output_hidden_states: If True, include hidden states in the output
            return_dict: If True, return a TokenClassifierOutput instead of a tuple

        Returns:
            If labels are provided, returns a tuple or TokenClassifierOutput containing:
                - loss: Computed cross-entropy loss for token classification
                - logits: Raw output scores for each token
                - hidden_states: (Optional) Hidden states from the model
                - attentions: (Optional) Attention weights from the model
        """
        # Determine whether to return a dictionary or tuple based on configuration
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Pass inputs through the base LiLT model
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

        # Extract the sequence output (hidden states for each token)
        sequence_output = outputs[0]

        # Compute logits using the token classification head
        logits = self.token_classifier_head(sequence_output)

        loss = None
        if labels is not None:
            # Ensure labels are on the same device as logits
            labels = labels.to(logits.device)
            # Define the cross-entropy loss function for classification
            loss_fct = nn.CrossEntropyLoss()
            # Compute the loss by reshaping logits and labels
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        # If not returning a dictionary, pack outputs into a tuple
        if not return_dict:
            output = (logits,) + outputs[2:]
            # Prepend loss to the output tuple if it was computed
            return ((loss,) + output) if loss is not None else output

        # Return a TokenClassifierOutput object containing loss, logits, hidden states, and attentions
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )