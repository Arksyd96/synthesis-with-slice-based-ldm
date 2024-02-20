import os
import torch
import pytorch_lightning as pl
import wandb
from torchvision.utils import save_image

def format_condition(arr):
    keys = ['voxel_volume', 'surface_area', 'sphericity', 'x', 'y', 'z', 'w', 'h', 'd']
    c = {k: '{:.2f}'.format(v) for k, v in zip(keys, arr)}
    c = ' - '.join([f'{k}: {v}' for k, v in c.items()])
    return c

class ImageReconstructionLogger(pl.Callback):
    def __init__(
        self, 
        n_samples = 5,
        sample_every_n_steps = 1,
        save = True, 
        save_dir = os.path.curdir,
        **kwargs
    ) -> None:
        super().__init__()
        self.n_samples = n_samples
        self.save = save
        self.sample_every_n_steps = sample_every_n_steps
        self.save_dir = '{}/images'.format(save_dir)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)

    def on_train_batch_end(self, trainer: pl.Trainer, pl_module, outputs, batch, batch_idx) -> None:
        # sampling only from master node master process each N iterations
        if trainer.global_rank == 0 and (trainer.global_step) % self.sample_every_n_steps == 0:
            pl_module.eval()
            
            with torch.no_grad():    
                for dataset, split in zip(
                    [trainer.train_dataloader.dataset, trainer.val_dataloaders.dataset], 
                    ['train', 'val']
                ):
                    batch = dataset.sample(self.n_samples)
                    
                    x = batch[0]
                    x = x.to(pl_module.device, torch.float32)

                    if pl_module.time_embedder is None:
                        pos = None
                    
                    x_hat, _, _ = pl_module(x, timestep=pos)
                    
                    spatial_stack = lambda x: torch.cat([
                        torch.hstack([img for img in x[:, idx, ...]]) for idx in range(x.shape[1])
                    ], dim=0)

                    args = [spatial_stack(arg) for arg in [x, x_hat]]
                    img = torch.cat(args, dim=0)

                    # [-1, 1] => [0, 255]
                    img = img.add(1).div(2).mul(255).clamp(0, 255).to(torch.uint8)
                    
                    wandb.log({
                        'Reconstruction examples': wandb.Image(
                            img.detach().cpu().numpy(), 
                            caption='{} - {} (Top are originals)'.format(split, trainer.current_epoch)
                        )
                    })
                    
                    if self.save:
                        x, x_hat = x.reshape(-1, 1, *x.shape[2:]), x_hat.reshape(-1, 1, *x_hat.shape[2:])
                        images = torch.cat([x, x_hat], dim=0)
                        save_image(images, '{}/sample_{}_{}.png'.format(self.save_dir, split, trainer.current_epoch), nrow=x.shape[0],
                                   normalize=True)


                        
class ImageGenerationLogger(pl.Callback):
    def __init__(
        self,
        noise_shape,
        save_every_n_epochs = 5,
        save = True,
        save_dir = os.path.curdir
    ) -> None:
        super().__init__()
        self.save = save
        self.save_every_n_epochs = save_every_n_epochs
        self.noise_shape = noise_shape
        self.save_dir = '{}/images'.format(save_dir)

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)

    def on_train_epoch_end(self, trainer, pl_module):
        pl_module.eval()
        if trainer.global_rank == 0 and (trainer.current_epoch + 1) % self.save_every_n_epochs == 0:
            with torch.no_grad():
                condition = trainer.train_dataloader.dataset.sample(1)[1]
                sample_img = pl_module.sample(num_samples=1, img_size=self.noise_shape, condition=condition).detach()

                if pl_module.std_norm is not None:    
                    sample_img = sample_img.mul(pl_module.std_norm)

                if pl_module.slice_based:
                    sample_img = sample_img.permute(0, 4, 1, 2, 3).squeeze(0)
                    sample_img = pl_module.latent_embedder.decode(sample_img, emb=None)
                else:
                    if pl_module.latent_embedder is not None:
                        sample_img = pl_module.latent_embedder.decode(sample_img, emb=None)
                    sample_img = sample_img.permute(0, 4, 1, 2, 3).squeeze(0)

                # selecting subset of the volume to display
                sample_img = sample_img[::4, ...] # 64 // 4 = 16

                if self.save:
                    save_image(sample_img[:, 0, None], '{}/sample_images_{}.png'.format(self.save_dir, trainer.current_epoch), normalize=True)

                sample_img = torch.cat([
                    torch.hstack([img for img in sample_img[:, idx, ...]]) for idx in range(sample_img.shape[1])
                ], dim=0)

                sample_img = sample_img.add(1).div(2).mul(255).clamp(0, 255).to(torch.uint8)
                
                wandb.log({
                    'Reconstruction examples': wandb.Image(
                        sample_img.cpu().numpy(), 
                        caption='{}'.format(trainer.current_epoch)
                    )
                })
