import os
import math
import torch
from torch import nn
from torch.distributions import RelaxedBernoulli, MultivariateNormal
import torch.nn.functional as F
from torchvision import models

from src.models.layers.base import MLP
from src.models.layers.base import get_layer_activation
from src.models.layers.intervention import maybe_intervene_scbm
# from utils.training import freeze_module, unfreeze_module


class SCBM(nn.Module):
    """
    Stochastic Concept Bottleneck Model (SCBM) with Learned Covariance Matrix.

    This class implements a Stochastic Concept Bottleneck Model (SCBM) that extends concept prediction by incorporating
    a learned covariance matrix. The SCBM aims to capture the uncertainty and dependencies between concepts, providing
    a more robust and interpretable model for concept-based learning tasks.

    Key Features:
    - Predicts concepts along with a learned covariance matrix to model the relationships and uncertainties between concepts.
    - Supports various training modes and intervention strategies to improve model performance and interpretability.

    Args:
        config (dict): Configuration dictionary containing model and data settings.

    Noteworthy Attributes:
        training_mode (str): The training mode (e.g., "joint", "sequential", "independent").
        num_monte_carlo (int): The number of Monte Carlo samples for uncertainty estimation.
        straight_through (bool): Flag indicating whether to use straight-through gradients.
        curr_temp (float): The current temperature for the Gumbel-Softmax distribution.
        cov_type (str): The type of covariance matrix ("empirical", "global", or "amortized", where "empirical is fixed at start").

    Methods:
        forward(x, epoch, validation=False, c_true=None):
            Perform a forward pass through the model.
        intervene(c_mcmc_probs, c_mcmc_logits):
            Perform an intervention on the model's concept predictions.
    """

    def __init__(self,
                input_size, 
                hidden_size, 
                output_size=2,
                decoder_hidden_size=64,
                n_layers_encoder=1,
                n_layers_decoder=1,
                decoder_type="mlp",
                activation='leaky_relu',
                cov_type="amortized",
                num_monte_carlo=100,
                training_mode="joint",
                straight_through=True,
                concept_learning="hard",
                max_epochs=100,
                alpha=1, # loss
                reg_precision="l1", # loss
                reg_weight=1, # loss
                interv_level=0.99,
                c_info={},
                y_info={}
        ):
        super(SCBM, self).__init__()

        # to be stored for every model
        self.has_concepts = True
        self.is_causal = False

        self.concept_names = c_info['names']
        self.concept_cardinality = c_info['cardinality']
        self.n_concepts = len(self.concept_names)
        self.n_concepts_binary = sum([1 if card == 2 else card for card in self.concept_cardinality])
        self.cov_type = cov_type

        # TODO: NOT IMPLEMENTED
        self.virtual_roots = [name for name in c_info['names'] if name.startswith('#virtual_')]

        self.training_epoch = 0
        self.num_monte_carlo = num_monte_carlo
        self.training_mode = training_mode  
        self.straight_through = straight_through     
        self.concept_learning = concept_learning

        self.curr_temp = 1.0
        self.max_epochs = max_epochs

        # loss params
        self.alpha = alpha if self.training_mode == "joint" else 1.0
        self.reg_precision = reg_precision
        self.reg_weight = reg_weight
        
        # intervention params
        self.interv_level = interv_level

        # Architectures
        # Encoder h(.)
        self.encoder = MLP(input_size=input_size,
                           hidden_size=hidden_size,
                           n_layers=n_layers_encoder,
                           activation=activation)
        
        # Mean of the concpet distribution
        self.mu_concepts = nn.Linear(hidden_size, 
                                     self.n_concepts_binary, 
                                     bias=True)

        # Covariance of the concept distribution
        if self.cov_type == "global":
            self.sigma_concepts = nn.Parameter(
                torch.zeros(int(self.n_concepts_binary * (self.n_concepts_binary + 1) / 2)))  # Predict lower triangle of concept covariance
        elif self.cov_type == "amortized":
            self.sigma_concepts = nn.Linear(hidden_size,
                                            int(self.n_concepts_binary * (self.n_concepts_binary + 1) / 2),
                                            bias=True)
            self.sigma_concepts.weight.data *= 0.01 # To prevent exploding precision matrix at initialization
        elif self.cov_type == "empirical": # not used
            self.sigma_concepts = torch.zeros(int(self.n_concepts_binary * (self.n_concepts_binary + 1) / 2))
        else:
            raise NotImplementedError(
                "Covariance type {} not implemented.".format(self.cov_type))

        # Assume binary concepts
        self.act_c = get_layer_activation('sigmoid')()

        if output_size is not None:
            if decoder_type == "linear":
                self.decoder = nn.Linear(self.n_concepts_binary, output_size)
            elif decoder_type == "mlp":
                self.decoder = MLP(input_size=self.n_concepts_binary,
                                    hidden_size=decoder_hidden_size,
                                    output_size=output_size,
                                    n_layers=n_layers_decoder,
                                    activation=activation)
            else:
                raise NotImplementedError(
                    "Decoder type {} not implemented.".format(decoder_type))
            

    def forward(self, x, c=None, intervention_index=None):
        """
        Perform a forward pass through the Stochastic Concept Bottleneck Model (SCBM).

        This method performs a forward pass through the SCBM, predicting concept probabilities and logits for the target variable.

        Args:
            x (torch.Tensor): The input features. Shape: (batch_size, input_dims)
            c (torch.Tensor, optional): The ground-truth concept values. Required for "independent" training mode. Default is None.
            intervention_index (torch.Tensor): Intervention index
            
        Returns:
            tuple: A tuple containing:
                - c_mcmc_prob (torch.Tensor): MCMC samples for predicted concept probabilities. Shape: (batch_size, num_concepts, num_monte_carlo)
                - c_triang_cov (torch.Tensor): Cholesky decomposition of the concept logit covariance matrix. Shape: (batch_size, num_concepts, num_concepts)
                - y_pred_logits (torch.Tensor): Logits for the target variable. Shape: (batch_size, num_classes)
                - c_mu (torch.Tensor, optional): Predicted concept means. Shape: (batch_size, num_concepts). Returned if `return_full` is True.
        Notes:
            - The method first obtains intermediate representations from the encoder.
            - It then predicts the concept means and the Cholesky decomposition of the covariance matrix in the logit space.
            - The method samples from the predicted normal distribution to obtain concept logits and probabilities.
            - Depending on the training mode, it handles different strategies for sampling and backpropagation.
            - Finally, it predicts the target variable logits by averaging over multiple Monte Carlo samples.
        """

        # custom inputs required by the original SCBM implementation
        epoch = self.training_epoch   # epoch (int): The current epoch number.
        # return_full = self.training  # return_full (bool, optional): Flag indicating whether to also return mu of concept. Default is True.

        # Get intermediate representations. This could be a CNN. 
        # In our implementation, we prerprocess the data with a CNN and therefore use only an MLP encoder here.
        intermediate = self.encoder(x)

        # Get mu and cholesky decomposition of covariance
        c_mu = self.mu_concepts(intermediate)
        if self.cov_type == "global":
            c_sigma = self.sigma_concepts.repeat(c_mu.size(0), 1)
        elif self.cov_type == "amortized":
            c_sigma = self.sigma_concepts(intermediate)
        elif self.cov_type == "empirical": # not used
            # c_sigma = self.sigma_concepts.unsqueeze(0).repeat(c_mu.size(0), 1, 1)
            raise NotImplementedError
        else:
            raise NotImplementedError(
                "Covariance type {} not implemented.".format(self.cov_type))

        if self.cov_type == "empirical":
            c_triang_cov = c_sigma
        else:
            # Fill the lower triangle of the covariance matrix with the values and make diagonal positive
            c_triang_cov = torch.zeros((c_sigma.shape[0], self.n_concepts_binary, self.n_concepts_binary), 
                                       device=c_sigma.device)
            rows, cols = torch.tril_indices(row=self.n_concepts_binary, 
                                            col=self.n_concepts_binary, 
                                            offset=0)
            diag_idx = rows == cols
            c_triang_cov[:, rows, cols] = c_sigma
            c_triang_cov[:, range(self.n_concepts_binary), range(self.n_concepts_binary)] = (
                F.softplus(c_sigma[:, diag_idx]) + 1e-6
            )

        # interventions, only at evaluation time
        has_interv = False
        if not self.training:
            if c is not None and intervention_index is not None:
                c_binary = self.from_multi_to_binary_tensor(c)
                intervention_index_binary = self.from_multi_to_binary_tensor_index(intervention_index)
                c_mcmc_prob, c_mcmc_logit, has_interv = maybe_intervene_scbm(c_mu, 
                                                                             c_triang_cov, 
                                                                             c_binary, 
                                                                             intervention_index_binary,
                                                                             self.num_monte_carlo,
                                                                             self.n_concepts_binary,
                                                                             self.interv_level)
        if not has_interv:
            # Sample from predicted normal distribution
            c_dist = MultivariateNormal(c_mu, scale_tril=c_triang_cov)
            # Sample from the distribution and reshape to [batch_size, num_concepts, mcmc_size]
            c_mcmc_logit = c_dist.rsample([self.num_monte_carlo]).movedim(0, -1)  
            c_mcmc_prob = self.act_c(c_mcmc_logit)

        # For all MCMC samples simultaneously sample from Bernoulli
        if not self.training:
            # No backpropagation necessary
            c_mcmc = torch.bernoulli(c_mcmc_prob)
        elif self.training_mode == "sequential":
            # No backpropagation necessary
            # c_mcmc = torch.bernoulli(c_mcmc_prob)
            raise NotImplementedError
        elif self.training_mode == "independent":
            # c_mcmc = c.unsqueeze(-1).repeat(1, 1, self.num_monte_carlo).float()
            raise NotImplementedError
        else:
            # Backpropagation necessary
            curr_temp = self.compute_temperature(epoch, 
                                                 device=c_mcmc_prob.device)
            dist = RelaxedBernoulli(temperature=curr_temp, 
                                    probs=c_mcmc_prob)

            # Bernoulli relaxation
            mcmc_relaxed = dist.rsample()
            if self.straight_through:
                # Straight-Through Gumbel Softmax
                mcmc_hard = (mcmc_relaxed > 0.5) * 1
                c_mcmc = mcmc_hard - mcmc_relaxed.detach() + mcmc_relaxed
            else:
                c_mcmc = mcmc_relaxed

        # MCMC loop for predicting label
        y_pred_probs_i = 0
        for i in range(self.num_monte_carlo):
            if self.concept_learning == "hard":
                c_i = c_mcmc[:, :, i]
            elif self.concept_learning == "soft":
                # c_i = c_mcmc_logit[:, :, i]
                raise NotImplementedError
            else:
                raise NotImplementedError
            y_pred_logits_i = self.decoder(c_i)
            y_pred_probs_i += torch.softmax(y_pred_logits_i, dim=1)

        # Average over MCMC samples
        y_pred_probs = y_pred_probs_i / self.num_monte_carlo
        y_pred_logits = torch.log(y_pred_probs + 1e-6)

        self.c_mu = c_mu
        self.c_triang_cov = c_triang_cov

        return [y_pred_logits, y_pred_probs], c_mcmc_prob
    
    def compute_temperature(self, epoch, device):
        final_temp = torch.tensor([0.5], device=device)
        init_temp = torch.tensor([1.0], device=device)
        rate = (math.log(final_temp) - math.log(init_temp)) / float(self.max_epochs)
        curr_temp = max(init_temp * math.exp(rate * epoch), final_temp)
        self.curr_temp = curr_temp
        return curr_temp
    
    def from_MCtensor_to_dict(self, input):
        output = {}
        i_init = 0
        for i, name in enumerate(self.concept_names):
            card = self.concept_cardinality[i]
            output[name] = torch.zeros(input.shape[0], card, input.shape[2], device=input.device)
            if card == 2:
                output[name][:,0,:] = 1 - input[:, i_init, :]
                output[name][:,1,:] = input[:, i_init, :]
                i_init += 1
            else:
                output[name] = input[:, i_init:i_init+card, :]
                i_init += card
            # average over MCMC samples
            output[name] = output[name].mean(dim=2)
        return output

    def from_multi_to_binary_tensor(self, input):
        output = [F.one_hot(input[:, i].long(), num_classes=self.concept_cardinality[i]) if self.concept_cardinality[i] > 2 
                  else input[:, i].unsqueeze(dim=-1)
                  for i in range(len(self.concept_names))]
        output = torch.cat(output, dim=-1)
        return output

    def from_multi_to_binary_tensor_index(self, input):
        output = [torch.ones(input.shape[0], self.concept_cardinality[i], device=input.device)*input[:, i, None] if self.concept_cardinality[i] > 2 
                  else input[:, i].unsqueeze(dim=-1)
                  for i in range(len(self.concept_names))]
        output = torch.cat(output, dim=-1)
        return output
            
    def filter_output_for_loss(self, y_output, c_output):
        """Filter output for loss function"""
        # output is [y_pred_logits, y_pred_probs], c_mcmc_prob
        y_pred_logits = y_output[0] 
        c_mcmc_prob = c_output
        return y_pred_logits, c_mcmc_prob
    
    def filter_output_for_metric(self, y_output, c_output):
        """Filter output for metric function"""
        # output is [y_pred_logits, y_pred_probs], c_mcmc_prob
        y_pred_probs = y_output[1]
        c_pred_probs = self.from_MCtensor_to_dict(c_output)
        return y_pred_probs, c_pred_probs

    def loss(self, y_pred_logits, y, c_mcmc_prob, c,
            # c_triang_cov: Tensor,
            # cov_not_triang=False
        ):
        """
        Compute the loss.

        Args:
            y_pred_logits (Tensor): Predicted target logits.
            y (Tensor): Ground-truth target values.
            c_mcmc_prob (Tensor): MCMC matrix of predicted concept probabilities.
            c (Tensor): Ground-truth concept values.

        Returns:
            Tensor: total loss.
        """
        # our c is in the multiclass space, so we need to convert it to binary
        target_true = y
        concepts_true = self.from_multi_to_binary_tensor(c)

        cov_not_triang=False

        assert torch.all((concepts_true == 0) | (concepts_true == 1))
        concepts_true_expanded = concepts_true.unsqueeze(-1).expand_as(
            c_mcmc_prob
        )

        bce_loss = F.binary_cross_entropy(
            c_mcmc_prob, concepts_true_expanded.float(), reduction="none"
        )  # [B,C,MCMC]
        intermediate_concepts_loss = -torch.sum(bce_loss, dim=1)  # [B,MCMC]
        mcmc_loss = -torch.logsumexp(
            intermediate_concepts_loss, dim=1
        )  # [B], logsumexp for numerical stability due to shift invariance
        concepts_loss = torch.mean(mcmc_loss)

        # if self.num_classes == 2:
        #     # Logits to probs
        #     target_pred_probs = nn.Sigmoid()(y_pred_logits.squeeze(1))
        #     target_loss = F.binary_cross_entropy(
        #         target_pred_probs, target_true.float(), reduction="mean"
        #     )
        # else:
        target_loss = F.cross_entropy(
            y_pred_logits, target_true.flatten().long(), reduction="mean"
        )

        # Add precision loss
        if self.reg_precision == "l1":
            if cov_not_triang:
                prec_matrix = torch.inverse(self.c_triang_cov)
            else:
                c_triang_inv = torch.inverse(self.c_triang_cov)
                prec_matrix = torch.matmul(
                    torch.transpose(c_triang_inv, dim0=1, dim1=2), c_triang_inv
                )
            prec_loss = prec_matrix.abs().sum(dim=(1, 2)) - prec_matrix.diagonal(
                offset=0, dim1=1, dim2=2
            ).abs().sum(-1)
            if prec_matrix.size(1) > 1:
                prec_loss = prec_loss / (
                    prec_matrix.size(1) * (prec_matrix.size(1) - 1)
                )
            else:  # Univariate case, can happen when intervening
                prec_loss = prec_loss
            prec_loss = prec_loss.mean(-1)
        else:
            prec_loss = torch.zeros_like(concepts_loss)

        total_loss = target_loss + self.alpha * concepts_loss + self.reg_weight * prec_loss

        return total_loss
    



    # def freeze_c(self):
    #     self.head.apply(freeze_module)

    # def freeze_t(self):
    #     self.head.apply(unfreeze_module)
    #     self.encoder.apply(freeze_module)
    #     self.mu_concepts.apply(freeze_module)
    #     if isinstance(self.sigma_concepts, nn.Linear):
    #         self.sigma_concepts.apply(freeze_module)
    #     else:
    #         self.sigma_concepts.requires_grad = False