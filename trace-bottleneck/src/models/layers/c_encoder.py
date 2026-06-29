import torch
import torch.nn as nn
from src.models.layers.base import Dense, MLP
from src.models.layers.intervention import maybe_intervene


class ConceptBlock(nn.Module):
    """
    Computes the concept embeddings and the concept probabilities from an input X.
    Args:
        input_size: int, size of the input tensor
        hidden_size: int, size of the hidden layer
        n_layers: int, number of layers in the MLP
        activation: str, activation function to use
        c_cardinality: int, number of possible values for each concept
    """
    def __init__(self, 
                    input_size, 
                    hidden_size,
                    n_layers=1,
                    activation='leaky_relu',
                    c_cardinality=2):
        super(ConceptBlock, self).__init__()

        # store concept information
        self.c_cardinality = c_cardinality
        self.hidden_size = hidden_size
        self.c_encoder = MLP(input_size=input_size,
                            hidden_size=c_cardinality*hidden_size,
                            # output_size=c_cardinality*hidden_size,
                            n_layers=n_layers,
                            activation=activation)
        self.c_scorer = nn.Linear(c_cardinality*hidden_size,
                                  c_cardinality)
        
    def forward(self, x, c_gt=None, intervention_index=None, to_return=['embs', 'probs']):
        """
        Returns:
            c_embedding: torch.Tensor, (batch_size, hidden_size) concept embeddings
            c_hat_probs: torch.Tensor, (batch_size, c_cardinality) concept probabilities
        """
        # encode the input tensor into concept embeddings
        c_values_embedding = self.c_encoder(x)

        # compute the logits and probabilities
        c_logits = self.c_scorer(c_values_embedding)
        if self.c_cardinality > 1:
            c_hat_probs = torch.softmax(c_logits, dim=1)
        else:
            # For regression (cardinality 1), output stays unbounded to match z-scored targets
            c_hat_probs = c_logits
        # Maybe intervene on concepts probabilities
        if c_gt is not None and intervention_index is not None:
            c_hat_probs = maybe_intervene(c_hat_probs, c_gt, intervention_index)
        
        # compute the weighted average between concepts embedding and probabilities
        first = c_values_embedding.reshape(-1, self.c_cardinality, self.hidden_size)
        second = c_hat_probs.unsqueeze(-1)
        c_embedding = (first * second).sum(dim=1)

        out = []
        if 'embs' in to_return:
            out.append(c_embedding)
        if 'probs' in to_return:
            out.append(c_hat_probs)
        if 'values_embs' in to_return:
            out.append(c_values_embedding)
        return tuple(out) if len(out) > 1 else out[0]