import torch
from torchmetrics import Metric
from torchmetrics.utilities.checks import _check_same_shape


def _cat_state(state):
    """Handle both list-of-tensors (single GPU) and already-concatenated tensor (post-DDP sync)."""
    if isinstance(state, torch.Tensor):
        return state
    if len(state) == 0:
        return torch.empty(0, dtype=torch.long)
    return torch.cat(state)

class ClassificationAccuracy(Metric):
    """
    Classification Accuracy is a standard metric that measures the proportion of correct predictions
    made by the model.
    """
    def __init__(self):
        super().__init__()
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, 
               preds: torch.Tensor, 
               target: torch.Tensor):
        # Manage Monte Carlo approximation tensor
        if len(preds.shape)>2: # shape = (n_samples, n_classes, n_mc_samples)
            preds = preds.mean(dim=-1)
        preds = preds.argmax(dim=-1)
        target = target.flatten().long()
        _check_same_shape(preds, target)
        correct = preds.eq(target).sum()
        self.correct += correct
        self.total += target.numel()

    def compute(self):
        return self.correct.float() / self.total


class WeightedAccuracy(Metric):
    """Inverse-class-frequency weighted accuracy over the accumulated epoch."""
    is_differentiable = False
    higher_is_better = True
    full_state_update = False

    def __init__(self):
        super().__init__()
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")

    def update(self,
               preds: torch.Tensor,
               target: torch.Tensor):
        if len(preds.shape) > 2:
            preds = preds.mean(dim=-1)
        preds = preds.argmax(dim=-1).long().flatten()
        target = target.flatten().long()
        _check_same_shape(preds, target)
        self.preds.append(preds)
        self.targets.append(target)

    def compute(self):
        preds = _cat_state(self.preds)
        targets = _cat_state(self.targets)
        if preds.numel() == 0:
            return torch.tensor(0.0)
        classes, counts = torch.unique(targets, return_counts=True)
        weights = torch.zeros_like(targets, dtype=torch.float32)
        for cls, count in zip(classes, counts):
            weights[targets == cls] = 1.0 / count.float().clamp(min=1.0)
        correct = preds.eq(targets).float()
        return (correct * weights).sum() / weights.sum().clamp(min=1e-8)


class RegressionMAE(Metric):
    """Mean Absolute Error for continuous (cardinality=1) concepts.
    Same .update(preds, target) API as ClassificationAccuracy."""
    def __init__(self):
        super().__init__()
        self.add_state("sum_ae", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # preds may be [B, 1] for card=1 concepts
        preds = preds.squeeze(-1).float() if preds.dim() > 1 else preds.float()
        target = target.flatten().float()
        self.sum_ae += (preds - target).abs().sum()
        self.total += target.numel()

    def compute(self):
        return self.sum_ae / self.total.clamp(min=1)
from sklearn.metrics import f1_score, precision_score, recall_score

class MacroF1(Metric):
    is_differentiable = False
    higher_is_better = True
    full_state_update = False

    def __init__(self):
        super().__init__()
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        if len(preds.shape) > 2:
            preds = preds.mean(dim=-1)
        preds = preds.argmax(dim=-1).long().flatten()
        target = target.flatten().long()
        self.preds.append(preds)
        self.targets.append(target)

    def compute(self):
        try:
            p = _cat_state(self.preds).cpu().numpy()
            t = _cat_state(self.targets).cpu().numpy()
            if len(p) == 0:
                return torch.tensor(0.0)
            val = f1_score(t, p, average="macro", zero_division=0)
            return torch.tensor(val, dtype=torch.float32)
        except Exception:
            return torch.tensor(0.0)

class MacroPrecision(Metric):
    is_differentiable = False
    higher_is_better = True
    full_state_update = False

    def __init__(self):
        super().__init__()
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        if len(preds.shape) > 2:
            preds = preds.mean(dim=-1)
        preds = preds.argmax(dim=-1).long().flatten()
        target = target.flatten().long()
        self.preds.append(preds)
        self.targets.append(target)

    def compute(self):
        try:
            p = _cat_state(self.preds).cpu().numpy()
            t = _cat_state(self.targets).cpu().numpy()
            if len(p) == 0:
                return torch.tensor(0.0)
            val = precision_score(t, p, average="macro", zero_division=0)
            return torch.tensor(val, dtype=torch.float32)
        except Exception:
            return torch.tensor(0.0)

class MacroRecall(Metric):
    is_differentiable = False
    higher_is_better = True
    full_state_update = False

    def __init__(self):
        super().__init__()
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        if len(preds.shape) > 2:
            preds = preds.mean(dim=-1)
        preds = preds.argmax(dim=-1).long().flatten()
        target = target.flatten().long()
        self.preds.append(preds)
        self.targets.append(target)

    def compute(self):
        try:
            p = _cat_state(self.preds).cpu().numpy()
            t = _cat_state(self.targets).cpu().numpy()
            if len(p) == 0:
                return torch.tensor(0.0)
            val = recall_score(t, p, average="macro", zero_division=0)
            return torch.tensor(val, dtype=torch.float32)
        except Exception:
            return torch.tensor(0.0)


def residual_concept_causal_effect(cace_metric_before, cace_metric_after):
    """
    Compute the residual concept causal effect between two concepts.
    Args:
        cace_metric_before: ConceptCausalEffect metric before the do-intervention on the inner concept
        cace_metric_after: ConceptCausalEffect metric after do-intervention on the inner concept
    """
    cace_before = cace_metric_before.compute()
    cace_after = cace_metric_after.compute()
    return cace_after / cace_before


class ConceptCausalEffect(Metric):
    """
    Concept Causal Effect (CaCE) is a metric that measures the causal effect between concept pairs
    or between a concept and the task.
    NOTE: only works on binary concepts.
    """
    def __init__(self):
        super().__init__()
        self.add_state("preds_do_1", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("preds_do_0", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, 
               preds_do_1: torch.Tensor, 
               preds_do_0: torch.Tensor):
        _check_same_shape(preds_do_1, preds_do_0)
        # expected value = 1*p(output=1|do(1)) + 0*(1-p(output=1|do(1))
        self.preds_do_1 += preds_do_1[:,1].sum()
        # expected value = 1*p(output=1|do(0)) + 0*(1-p(output=1|do(0))
        self.preds_do_0 += preds_do_0[:,1].sum()
        self.total += preds_do_1.size()[0]

    def compute(self):
        return (self.preds_do_1.float() / self.total) - (self.preds_do_0.float()  / self.total)




def edge_type(graph, i, j):
    if graph[i,j]==1 and graph[j,i]==0:
        return 'i->j'
    elif graph[i,j]==0 and graph[j,i]==1:
        return 'i<-j'
    elif (graph[i,j]==-1 and graph[j,i]==-1) or (graph[i,j]==1 and graph[j,i]==1):
        return 'i-j'
    elif graph[i,j]==0 and graph[j,i]==0:
        return '/'
    else:
        raise ValueError(f'invalid edge type {i}, {j}')

# graph similairty metrics
def hamming_distance(first, second):
    """Compute the graph edit distance between two partially direceted graphs"""
    first = first.loc[[row for row in first.index if '#virtual_' not in row],
                      [col for col in first.columns if '#virtual_' not in col]]
    first = torch.Tensor(first.values)
    second = second.loc[[row for row in second.index if '#virtual_' not in row],
                        [col for col in second.columns if '#virtual_' not in col]]
    second = torch.Tensor(second.values)
    assert (first.diag() == 0).all() and (second.diag() == 0).all()
    assert first.size() == second.size()
    N = first.size(0)
    cost = 0
    count = 0
    for i in range(N):
        for j in range(i, N):
            if i==j: continue
            if edge_type(first, i, j)==edge_type(second, i, j): continue
            else:
                count += 1
                # edge was directed
                if edge_type(first, i, j)=='i->j' and edge_type(second, i, j)=='/': cost += 1./4.
                elif edge_type(first, i, j)=='i<-j' and edge_type(second, i, j)=='/': cost += 1./4.
                elif edge_type(first, i, j)=='i->j' and edge_type(second, i, j)=='i-j': cost += 1./5.
                elif edge_type(first, i, j)=='i<-j' and edge_type(second, i, j)=='i-j': cost += 1./5.
                elif edge_type(first, i, j)=='i->j' and edge_type(second, i, j)=='i<-j': cost += 1./3.
                elif edge_type(first, i, j)=='i<-j' and edge_type(second, i, j)=='i->j': cost += 1./3.
                # edge was undirected
                elif edge_type(first, i, j)=='i-j' and edge_type(second, i, j)=='/': cost += 1./4.
                elif edge_type(first, i, j)=='i-j' and edge_type(second, i, j)=='i->j': cost += 1./4. 
                elif edge_type(first, i, j)=='i-j' and edge_type(second, i, j)=='i<-j': cost += 1./4.
                # there was no edge
                elif edge_type(first, i, j)=='/' and edge_type(second, i, j)=='i-j': cost += 1./2.
                elif edge_type(first, i, j)=='/' and edge_type(second, i, j)=='i->j': cost += 1
                elif edge_type(first, i, j)=='/' and edge_type(second, i, j)=='i<-j': cost += 1

                else:  
                    raise ValueError(f'invalid combination of edge types {i}, {j}')
    
    # cost = cost / (N*(N-1))/2
    return cost, count