import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from datetime import datetime

import torch 
import pytorch_lightning as pl
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import wandb as wandb_logger

from modules.data_preprocessing import BRATSDataModule
from modules.models.autoencoders import VAE

os.environ['WANDB_API_KEY'] = 'your wandb api key'

if __name__ == "__main__":
    pl.seed_everything(42)

    # --------------- Settings --------------------
    current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    save_dir = '{}/runs/first-stage-{}'.format(os.path.curdir, str(current_time))
    os.makedirs(save_dir, exist_ok=True)

    # --------------- Logger --------------------
    logger = wandb_logger.WandbLogger(
        project     = 'slice-based-latent-diffusion-model', 
        name        = 'first-stage training',
        save_dir    = save_dir
    )

    # ------------ Load Data ----------------
    datamodule = BRATSDataModule(
        train_dir       = './data/first_stage_dataset_192x192.npy',
        train_ratio     = 0.95,
        norm            = 'centered-norm', 
        batch_size      = 32,
        num_workers     = 32,
        shuffle         = True,
        # horizontal_flip = 0.2,
        # vertical_flip   = 0.2,
        # rotation        = (0, 90),
        # random_crop_size = (96, 96),
        dtype           = torch.float32,
        include_radiomics = False
    )

    # ------------ Initialize Model ------------
    # model = VAE.load_from_checkpoint('./runs/first-stage-2024_01_12_142022/last.ckpt')

    model = VAE(
        in_channels     = 2, 
        out_channels    = 2, 
        emb_channels    = 4,
        spatial_dims    = 2, # 2D or 3D
        hid_chs         = [128, 256, 384, 512], 
        kernel_sizes    = [3, 3, 3, 3],
        strides         = [1, 2, 2, 2],
        use_attention   = [False, False, False, True],
        time_embedder   = None,
        deep_supervision = False,
        embedding_loss_weight = 1e-6,
        precision = 32
    )

    # -------------- Training Initialization ---------------
    checkpointing = ModelCheckpoint(
        dirpath     = save_dir, # dirpath
        monitor     = 'val/ssim', # 'val/ae_loss_epoch',
        every_n_epochs = 1,
        save_last   = True,
        save_top_k  = 1,
        mode        = 'max',
    )
        
    trainer = Trainer(
        logger      = logger,
        precision   = 32,
        accelerator = 'gpu',
        # gradient_clip_val=0.5,
        default_root_dir = save_dir,
        enable_checkpointing = True,
        check_val_every_n_epoch = 1,
        log_every_n_steps = 1, 
        min_epochs = 100,
        max_epochs = 10000,
        num_sanity_val_steps = 0,
        callbacks=[checkpointing]
    )
    
    # ---------------- Execute Training ----------------
    trainer.fit(model, datamodule=datamodule)

    # ------------- Save path to best model -------------
    model.save_best_checkpoint(trainer.logger.save_dir, checkpointing.best_model_path)
