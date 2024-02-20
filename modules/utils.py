import torch
from torch import nn

def kl_gaussians(mean1, logvar1, mean2, logvar2):
    """ Compute the KL divergence between two gaussians."""
    return 0.5 * (logvar2-logvar1 + torch.exp(logvar1 - logvar2) + torch.pow(mean1 - mean2, 2) * torch.exp(-logvar2)-1.0)

def spatially_stack_latents(latents, grid_size, index_channel=False):
    batch_size, num_latents, num_channels, height, width = latents.size()
    latents_stacked = latents.permute(0, 2, 1, 3, 4).contiguous()  # Rearrange dimensions
    
    grid_height = grid_size[0]
    grid_width = grid_size[1]
    
    num_cols = num_latents // grid_height
    num_rows = num_latents // grid_width
    
    assert num_cols * num_rows == num_latents, "Grid size is not compatible with number of latents"

    latents_stacked = latents_stacked.view(batch_size, num_channels, num_rows, num_cols, height, width)
    latents_stacked = latents_stacked.permute(0, 1, 2, 4, 3, 5).contiguous()  # Rearrange dimensions
    latents_stacked = latents_stacked.view(batch_size, num_channels, num_rows * height, num_cols * width)

    if index_channel:
        # generating an index channel
        index_channel = torch.ones(batch_size, 1, num_rows * height, num_cols * width)
        for i in range(num_rows):
            for j in range(num_cols):
                index_channel[:, :, i * height : (i + 1) * height, j * width : (j + 1) * width] *= (i * num_cols + j)
        
        # Add index channel
        latents_stacked = torch.cat([latents_stacked, index_channel.to(latents_stacked.device)], dim=1)
    
    return latents_stacked

def reverse_spatial_stack(latents_stacked, shape, index_channel=False):
    if index_channel: # we drop the index channel
        latents_stacked = latents_stacked[:, :-1, ...]

    batch_size, num_channels, height, width = latents_stacked.size()
    orig_height, orig_width = shape
    num_cols = width // orig_width
    num_rows = height // orig_height

    latents_stacked = latents_stacked.reshape(batch_size, num_channels, num_rows, orig_height, num_cols, orig_width)
    latents_stacked = latents_stacked.permute(0, 1, 2, 4, 3, 5).contiguous()  # Rearrange dimensions
    latents_stacked = latents_stacked.reshape(batch_size, num_channels, num_rows * num_cols, orig_height, orig_width)
    
    # channels first
    latents_stacked = latents_stacked.permute(0, 2, 1, 3, 4).contiguous()  # Rearrange dimensions

    return latents_stacked

def add_index_channel(latents_stacked, grid_size):
    batch_size, num_channels, height, width = latents_stacked.size()
    num_cols = grid_size[0]
    num_rows = grid_size[1]
    
    height = height // num_cols
    width = width // num_rows
    
    assert height % num_cols == 0, f'Height {height} is not divisible by grid height {num_cols}'
    assert width % num_rows == 0, f'Width {width} is not divisible by grid width {num_rows}'

    # generating an index channel
    index_channel = torch.ones(batch_size, 1, num_rows * height, num_cols * width)
    for i in range(num_rows):
        for j in range(num_cols):
            index_channel[:, :, i * height : (i + 1) * height, j * width : (j + 1) * width] *= (i * num_cols + j)
    
    # Add index channel
    latents_stacked = torch.cat([latents_stacked, index_channel.to(latents_stacked.device)], dim=1)

    return latents_stacked
