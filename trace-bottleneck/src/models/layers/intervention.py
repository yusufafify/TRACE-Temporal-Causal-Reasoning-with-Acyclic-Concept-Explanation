import torch
from torch.nn.functional import one_hot
# for scbm
from src.models.scbm_utils.intervention_scbm import SCBM_Strategy
from src.models.scbm_utils.utils import numerical_stability_check

def get_test_intervention_index(c_shape, c_index, values=None):   
    """
    Get intervention index for test time intervention.
    Args:
        c_shape: shape of the concept tensor
        c_index: (int or list[int]) indices of concepts to intervene on
        values: (int or Tensor or str) or list of (int or Tensor or str), if None, it only return the intervention index
                                                                          if int, set all intervened concepts to this value
                                                                          if Tensor, set intervened concepts to this tensor
                                                                          if 'random', set intervened concepts to random values
    """       
    intervention_index = torch.zeros(c_shape)
    # c_values = torch.full(c_shape, float('nan'))
    c_values = torch.full(c_shape, -99999999.)
    if isinstance(c_index, int):
        c_index = [c_index]
    if values is not None and not isinstance(values, list):
        values = [values]

    if c_index:
        for i in range(len(c_index)):
            # indices
            index_i = c_index[i]
            intervention_index[:,index_i] = 1
            # values
            if values is not None:
                values_i = values[i]
                if isinstance(values_i, int):
                    c_values[:, index_i] = torch.ones(c_shape[0]) * values_i
                elif isinstance(values, torch.Tensor):
                    c_values[:, index_i] = values_i
                elif values == 'random':
                    raise NotImplementedError
                
    if values is not None:
        return intervention_index.to("cuda" if torch.cuda.is_available() else "cpu"), c_values.to("cuda" if torch.cuda.is_available() else "cpu")
    else:
        return intervention_index.to("cuda" if torch.cuda.is_available() else "cpu")

def maybe_intervene(c_pred_probs, c, intervention_index):
    # check if intervention index is not all zeros (non interventions) and ground truth is not all nans (virutal roots)
    if not torch.all(intervention_index == 0) and not torch.all(torch.isnan(c)):
        # check if c is nan where intervention index is not 0
        if torch.any(torch.isnan(c) & (intervention_index != 0)):
            raise ValueError("Intervention with nan ground truth is not allowed")
        concept_cardinality = c_pred_probs.shape[1]
        if concept_cardinality == 1:
            # Regression concept: directly replace predicted value with ground truth
            index = intervention_index.bool().unsqueeze(1)
            c_gt = c.float().unsqueeze(1) if c.dim() == 1 else c.float()
            c_pred_probs = torch.where(index, c_gt, c_pred_probs)
        else:
            # Classification concept: replace with one-hot ground truth
            index = intervention_index.bool().unsqueeze(1).repeat(1,concept_cardinality)
            # Binarize continuous concept values for intervention (threshold 0.5)
            c_bin = c.round().long().clamp(0, concept_cardinality - 1)
            c_one_hot = one_hot(c_bin, concept_cardinality).float()
            c_pred_probs = torch.where(index, c_one_hot, c_pred_probs)
    return c_pred_probs



def maybe_intervene_scbm(c_mu, c_triang_cov, c, intervention_index,
                         num_monte_carlo, n_concepts_binary, interv_level):
    # check if intervention index is not all zeros (non interventions) and ground truth is not all nans (virutal roots)
    if not torch.all(intervention_index == 0) and not torch.all(torch.isnan(c)):
        # check if c is nan where intervention index is not 0
        if torch.any(torch.isnan(c) & (intervention_index != 0)) or torch.any((c==-99999999.) & (intervention_index != 0)):
            raise ValueError("Intervention with nan ground truth is not allowed")
        
        # handle c_cov
        c_triang_cov = c_triang_cov.to(torch.float64)
        c_mu = c_mu.to(torch.float64)
        c_cov = torch.matmul(
            c_triang_cov,
            torch.transpose(c_triang_cov, dim0=1, dim1=2),
        )
        c_cov = numerical_stability_check(c_cov, device=c_cov.device)
        
        # perform intervention
        with torch.inference_mode(False):
            interv_str = SCBM_Strategy(inter_strategy='conf_interval_optimal', 
                                        train_loader=None, # not used for 'conf_interval_optimal'
                                        model=None, 
                                        device=None, 
                                        num_monte_carlo=num_monte_carlo,
                                        num_concepts=n_concepts_binary,
                                        level=interv_level)
            output = interv_str.compute_intervention(c_mu=c_mu, 
                                                    c_cov=c_cov, 
                                                    c_true=c, 
                                                    c_mask=intervention_index)
            c_interv_mu, c_interv_cov, c_mcmc_prob, c_mcmc_logit = output
        return c_mcmc_prob, c_mcmc_logit, True
    else:
        return None, None, False