import torch
import torch.nn as nn

class EikonalLoss(nn.Module):
    def __init__(self):
        super(EikonalLoss, self).__init__()

    def forward(self, sdf_output, locations):
        gradients = torch.autograd.grad(
            outputs=sdf_output,
            inputs=locations,
            grad_outputs=torch.ones_like(sdf_output),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
            allow_unused=True  # Allow unused tensors
        )[0]
       
        if gradients is None:
            return torch.tensor(0.0, device=sdf_output.device)  # Return zero loss if gradients are None
       
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1))
        eikonal_loss = torch.mean((gradients_norm - 1) ** 2)
        return eikonal_loss

class ConfidenceLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=5.0):
        super(ConfidenceLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, preds, targets):
        non_one_mask = (targets != 1).float()
        squared_error = (preds - targets) ** 2
        loss = (self.alpha * (1 - non_one_mask) + self.beta * non_one_mask) * squared_error
        return loss.mean()