# Import necessary libraries


from typing import Optional, Union, Tuple # Import type hints
import torch # PyTorch library
import torch.nn as nn # Import PyTorch neural network module for building custom deep learning layers and mdels
from transformers.modeling_outputs import TokenClassifierOutput # Import the output class for token classification
from transformers import LiltPreTrainedModel, LiltModel # Import the Lilt core model and its pre-trained version (weights)

# Define a token classification head for LiLT model
class TokenClassificationHead(nn.Module):
    def __init__(self, config):
        """
        Initializes the token classification head

        Args:
            config: Model configuration containing parameters like dropout rates,
                    hidden size, and number of labels
        """
        super().__init__() # There is no need to call the parent class constructor here as nn.Module does not have any specific initialization (config)

        # Choose the classifier dropout from config, fallback to hidden dropout if not set
        classifier_dropout = config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        # Initialize dropout layer with the determined dropout probability
        self.dropout = nn.Dropout(classifier_dropout)

        # Define a linear layer to project hidden states to the number of labels
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        """
        Forward pass for the token classification head

        Args:
            features: Input features (e.g., hidden states) from the base model
            **kwargs: Additional arguments (not used in this head)

        Returns:
            logits: The output logits for each token
        """
        # Apply dropout to the features to prevent overfitting
        x = self.dropout(features)
        # Apply the linear classifier to obtain logits for each label
        logits = self.classifier(x)
        return logits


# Define the LiLT model for token classification by extending the pre-trained LiLT model
class LiLTForTokenClassification(LiltPreTrainedModel):
    def __init__(self, config):
        """
        Initializes the LiLT model for token classification

        Args:
            config: Model configuration containing parameters for LiLT and the classifier head
        """
        super().__init__(config)
        # Initialize the base LiLT model without the pooling layer
        self.lilt = LiltModel(config, add_pooling_layer=False)
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