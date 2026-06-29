import torch
import torch.nn as nn

from src.models.layers.base import MLP

class BlackBox(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size=2,
                 n_layers=1,
                 activation='leaky_relu',
                 dropout=0.0,
                 c_info={},
                 y_info={}):
        super(BlackBox, self).__init__()
        
        # to be stored for every model
        self.has_concepts = False
        self.is_causal = False

        self.mlp = MLP(input_size=input_size,
                           hidden_size=hidden_size,
                           output_size=output_size,
                           n_layers=n_layers,
                           activation=activation,
                           dropout=dropout)

    def forward(self, x, c=None, intervention_index=None):
        y_hat = self.mlp(x)
        y_hat_probs = torch.softmax(y_hat, dim=1)
        return y_hat_probs, None
    
    def filter_output_for_loss(self, y_output, c_output):
        return y_output, None
    
    def filter_output_for_metric(self, y_output, c_output):
        return y_output, None

    def loss(self, y_hat, y, c_hat_dict, c):
        """Compute loss function.
        Args:
            y_hat (torch.Tensor): Predicted task probabilities
            y (torch.Tensor): True task labels
            c_hat_dict (Dict): Predicted concept probabilities
            c (torch.Tensor): True concept labels"""
        y = y.flatten().long()
        # c = c.long() # later to avoid nan to disappear
        loss_form = torch.nn.NLLLoss()

        # -- task loss
        y_hat = torch.log(y_hat + 1e-6)
        task_loss = loss_form(y_hat, y)

        return task_loss

    # def loss_CE(self, y_hat, y, c_hat_dict, c):
    #     """Compute the loss function.
    #     Args:
    #         y_hat: Predicted logits.
    #         y: True labels.
    #         (unused) c_hat_dict: Predicted concept probabilities.
    #         (unused) c: True concept values."""
    #     y = y.flatten().long()
    #     # cross entropy
    #     return torch.nn.functional.cross_entropy(y_hat, y)