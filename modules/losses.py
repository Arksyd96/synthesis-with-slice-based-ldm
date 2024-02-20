import torch 
import torch.nn.functional as F 
import lpips

class LPIPS(torch.nn.Module):
    """Learned Perceptual Image Patch Similarity (LPIPS)"""
    def __init__(self, linear_calibration=False, normalize=False):
        super().__init__()
        self.loss_fn = lpips.LPIPS(net='vgg', lpips=linear_calibration) # Note: only 'vgg' valid as loss  
        self.normalize = normalize # If true, normalize [0, 1] to [-1, 1]
        

    def forward(self, pred, target):
        # No need to do that because ScalingLayer was introduced in version 0.1 which does this indirectly  
        # if pred.shape[1] == 1: # convert 1-channel gray images to 3-channel RGB
        #     pred = torch.concat([pred, pred, pred], dim=1)
        # if target.shape[1] == 1: # convert 1-channel gray images to 3-channel RGB 
        #     target = torch.concat([target, target, target], dim=1)

        if pred.ndim == 5: # 3D Image: Just use 2D model and compute average over slices 
            depth = pred.shape[2] 
            losses = torch.stack([self.loss_fn(pred[:,:,d], target[:,:,d], normalize=self.normalize) for d in range(depth)], dim=2)
            return torch.mean(losses, dim=2, keepdim=True)
        else:
            return self.loss_fn(pred, target, normalize=self.normalize)

def exp_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(torch.exp(-logits_real))
    loss_fake = torch.mean(torch.exp(logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss

def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss

def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(F.softplus(-logits_real)) +
        torch.mean(F.softplus(logits_fake)))
    return d_loss

def kl_gaussians(mean1, logvar1, mean2, logvar2):
    """ Compute the KL divergence between two gaussians."""
    return 0.5 * (logvar2 - logvar1 + torch.exp(logvar1 - logvar2) + torch.pow(mean1 - mean2, 2) * torch.exp(-logvar2)-1.0)



 