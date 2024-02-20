import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from datetime import datetime
import torch 
import pytorch_lightning as pl
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from pytorch_lightning.loggers import wandb as wandb_logger

from modules.diffusion import DiffusionPipeline
from modules.models.unet import UNet
from modules.models.autoencoders import TimeEmbbeding
from modules.scheduler import GaussianNoiseScheduler
from modules.loggers import ImageGenerationLogger
from modules.models.autoencoders import VAE, ConditionMLP

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from modules.data_preprocessing import BRATSDataModule

import os
os.environ['WANDB_API_KEY'] = 'your wandb api key'

if __name__ == "__main__":
    pl.seed_everything(42)
    torch.set_float32_matmul_precision('high')

    # --------------- Settings --------------------
    current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    save_dir = '{}/runs/second-stage-{}'.format(os.path.curdir, str(current_time))
    os.makedirs(save_dir, exist_ok=True)

    # --------------- Logger --------------------
    logger = wandb_logger.WandbLogger(
        project = 'slice-based-latent-diffusion-model', 
        name    = 'second-stage training',
        save_dir = save_dir
    )

    datamodule = BRATSDataModule(
        train_dir       = './data/second_stage_dataset_192x192.npy',
        train_ratio     = 1.0,
        norm            = 'centered-norm', 
        batch_size      = 2,
        num_workers     = 16,
        shuffle         = True,
        # horizontal_flip = 0.5,
        # vertical_flip   = 0.5,
        # rotation        = (0, 90),
        # random_crop_size = (96, 96),
        dtype           = torch.float32,
        include_radiomics = True
    )


    # ------------ Initialize Model ------------
    cond_embedder = ConditionMLP
    cond_embedder_kwargs = {
        'in_features': 9, 
        'out_features': 512, 
        'hidden_dim': 256
    }
 

    time_embedder = TimeEmbbeding
    time_embedder_kwargs = {
        'emb_dim': 512 # stable diffusion uses 4 * model_channels (model_channels is about 256)
    }


    noise_estimator = UNet
    noise_estimator_kwargs = {
        'in_ch': 4,
        'out_ch': 4,  
        'spatial_dims': 2,
        'hid_chs': [64, 128, 256, 512],
        'kernel_sizes': [3, 3, 3, 3],
        'strides': [1, 2, 2, 2],
        'time_embedder': time_embedder,
        'time_embedder_kwargs': time_embedder_kwargs,
        'cond_embedder': cond_embedder,
        'cond_embedder_kwargs': cond_embedder_kwargs,
        'deep_supervision': False,
        'use_res_block': True,
        'use_attention': 'none',
    }


    # ------------ Initialize Noise ------------
    noise_scheduler = GaussianNoiseScheduler
    noise_scheduler_kwargs = {
        'timesteps': 1000,
        'beta_start': 0.002, # 0.0001, 0.0015
        'beta_end': 0.02, # 0.01, 0.0195
        'schedule_strategy': 'scaled_linear'
    }
    
    # ------------ Initialize Latent Space  ------------
    latent_embedder_checkpoint = './runs/first_stage-2023_08_29_114132_vae_4_ch/last.ckpt'
    latent_embedder = VAE.load_from_checkpoint(latent_embedder_checkpoint)


    # ------------ Initialize Pipeline ------------
    pipeline = DiffusionPipeline(
        noise_estimator = noise_estimator, 
        noise_estimator_kwargs = noise_estimator_kwargs,
        noise_scheduler = noise_scheduler, 
        noise_scheduler_kwargs = noise_scheduler_kwargs,
        latent_embedder = latent_embedder,
        estimator_objective = 'x_T',
        estimate_variance = False, 
        use_self_conditioning = False, 
        use_ema = False,
        classifier_free_guidance_dropout = 0.0, # Disable during training by setting to 0
        clip_x0 = False,
        slice_based = True,
        std_norm = 0.895475213 # to be computed first with compute_latents_std.py
    )

    # -------------- Training Initialization ---------------
    checkpointing = ModelCheckpoint(
        dirpath = str(save_dir), # dirpath
        monitor = None,
        every_n_epochs = 50,
        save_last = True,
        save_top_k = 1
    )

    image_logger = ImageGenerationLogger(
        noise_shape = (6, 24, 24, 96), # to change if in 3D
        save_dir = str(save_dir),
        save_every_n_epochs = 20,
        save = True
    )

    trainer = Trainer(
        logger = logger,
        precision = 32,
        accelerator = 'gpu',
        default_root_dir = save_dir,
        enable_checkpointing = True,
        log_every_n_steps = 1, 
        min_epochs = 100,
        max_epochs = 10000,
        num_sanity_val_steps = 0,
        # fast_dev_run = 10,
        callbacks=[checkpointing, image_logger]
    )
    
    
    # ---------------- Execute Training ----------------
    trainer.fit(pipeline, datamodule=datamodule)

    # ------------- Save path to best model -------------
    pipeline.save_best_checkpoint(trainer.logger.save_dir, checkpointing.best_model_path)


