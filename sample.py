import torch
import argparse
import sys
import numpy as np
from tqdm import tqdm

from modules.models.autoencoders import VAEGAN, VAE
from modules.diffusion import DiffusionPipeline

def replace_neg_pos(tensor, x, y):
    tensor[tensor < 0] = x
    tensor[tensor >= 0] = y
    return tensor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--model-path', type=str, required=True, help='Path to the diffusion model')
    parser.add_argument('-ac', '--ae-class', type=str, required=True, choices=['VAEGAN', 'VAE'], help='Class of the associated latent autoencoder')
    parser.add_argument('-ap', '--ae-path', type=str, required=True, help='Path to the associated latent autoencoder')
    parser.add_argument('-v', '--std-norm', type=float, default=1.0, help=
                        'Standard deviation of the encoded latents in order to denormalize sampled latents')
    parser.add_argument('-n', '--num-samples', type=int, default=1, help='Number of samples to generate')
    parser.add_argument('-s', '--noise-shape', type=int, nargs='+', default=[1, 16, 16], help='Shape of the noise to sample')
    parser.add_argument('--ddim', action='store_true', help='Whether to use DDIM or not', default=False)
    parser.add_argument('--ddim-steps', type=int, default=50, help='Number of DDIM steps')
    parser.add_argument('--condition', type=float, nargs='+', default=None, help='Condition to use for sampling [v_0, v_1, ..., v_n]')
    parser.add_argument('--save-volume', action='store_true', help='Save the generated volume')
    parser.add_argument('--save-png', action='store_true', help='Save the generated volume as sequence of PNGs')
    parser.add_argument('--save-nifti', action='store_true', help='Save the generated volume as Nifti')
    parser.add_argument('--save-path', type=str, default='./', help='Path to save the generated volume & PNG')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to use')
    args = parser.parse_args()

    # get a class from a string
    autoencoder = getattr(sys.modules[__name__], args.ae_class)
    autoencoder = autoencoder.load_from_checkpoint(args.ae_path, time_embedder=None)

    diffuser = DiffusionPipeline.load_from_checkpoint(args.model_path, latent_embedder=autoencoder, std_norm=args.std_norm)

    device = torch.device(args.device)
    autoencoder = autoencoder.to(device)
    diffuser = diffuser.to(device)
    diffuser.eval()

    if args.condition is not None:
        # generating conditions randomly, this is not a great idea since it can generate unrealistic conditions
        # continuous conditions is a drawback to our method, since it models a regression task which requires a lot more data
        if np.sum(args.condition) == 0:
            condition = torch.zeros(size=(args.num_samples, 9), dtype=torch.float32, device=device)
            for idx in range(args.num_samples):
                voxel_volume = np.random.uniform(low=0.01, high=0.9)
                surface_area = voxel_volume * (2/3)
                sphericity = np.random.uniform(low=0.25, high=0.9)
                x, y, z, w, h, d = np.random.uniform(low=0.01, high=0.99, size=6)
                condition[idx, :] = torch.tensor(
                    [voxel_volume, surface_area, sphericity, x, y, z, w, h, d], 
                    dtype=torch.float32, device=device
                )
        else:
            condition = [torch.tensor(args.condition, dtype=torch.float32, device=device).unsqueeze(0)]
    else:
        condition = None

    samples = []
    with torch.no_grad():
        for idx in tqdm(range(args.num_samples), position=0, leave=True): # sampling with a batch size of 1 for memory issues
            volume = diffuser.sample(
                num_samples=1, 
                img_size=args.noise_shape,
                condition=(condition[idx, None] if condition.__len__() > 1 else condition[0]) if args.condition is not None else None,
                use_ddim=args.ddim,
                steps=args.ddim_steps if args.ddim else 1000
            ).detach()

            volume = volume.mul(torch.tensor(args.std_norm, dtype=torch.float32, device=device))

            if args.slice_based:
                # slice-based latent diffusion; permute depth to batch for decoding
                volume = volume.permute(0, 4, 1, 2, 3).squeeze(0)

            # decode D slices at once
            volume = diffuser.latent_embedder.decode(volume, emb=torch.arange(args.noise_shape[-1]).reshape(-1, 1))
            
            if args.slice_based:
                volume = volume.unsqueeze(0)
            
            samples.append(volume)
        
        samples = torch.cat(samples, dim=0)
        if args.slice_based:
            samples = samples.permute(0, 2, 3, 4, 1)

        print('Samples shape: {}'.format(samples.shape))

        samples[:, 1] = replace_neg_pos(samples[:, 1], -1, 1) # rounding to -1, 1
        N, C, W, H, D = samples.shape

    if args.save_png:
        from torchvision.utils import save_image
        for idx in range(samples.shape[0]):
            s = samples[idx, ...].permute(3, 0, 1, 2)
            s = torch.vstack([s[:, c, ...] for c in range(s.shape[1])]).unsqueeze(1)
            save_image(s, '{}/sample_{}.png'.format(args.save_path, idx), nrow=D, normalize=True)

    if args.save_volume:
        import numpy as np
        np.save('{}/samples.npy'.format(args.save_path), samples.cpu().numpy())
    
    if args.save_nifti:
        import nibabel as nib
        for idx in range(args.num_samples):
            volume = samples[idx, 0, ...].cpu().numpy()
            nib.save(nib.Nifti1Image(volume, np.eye(4)), '{}/sample_{}.nii.gz'.format(args.save_path, idx))

    print('Done!')
