"""
    Code inspired by medfusion public repo : https://github.com/mueller-franzes/medfusion
    And official Open AI repo : https://github.com/openai/guided-diffusion (https://arxiv.org/abs/2212.07501)
    The code is modified to fit the needs of the project
"""

from pathlib import Path 

import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torchvision.utils import save_image

from modules.models.base import DownBlock, UpBlock, BasicBlock, BasicResBlock, UnetResBlock, UnetBasicBlock
from modules.losses import hinge_d_loss, LPIPS
from modules.models.base import BasicModel, VeryBasicModel
import wandb
from monai.networks.layers.utils import get_act_layer

from pytorch_msssim import ssim
import math 


class SinusoidalPosEmb(nn.Module):
    def __init__(self, emb_dim=16, downscale_freq_shift=1, max_period=10000, flip_sin_to_cos=False):
        super().__init__()
        self.emb_dim = emb_dim
        self.downscale_freq_shift = downscale_freq_shift
        self.max_period = max_period
        self.flip_sin_to_cos=flip_sin_to_cos

    def forward(self, x):
        device = x.device
        half_dim = self.emb_dim // 2
        emb = math.log(self.max_period) / (half_dim - self.downscale_freq_shift)
        emb = torch.exp(-emb*torch.arange(half_dim, device=device))
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)

        if self.flip_sin_to_cos:
            emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)
        
        if self.emb_dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
        return emb


class LearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with learned sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, emb_dim):
        super().__init__()
        self.emb_dim = emb_dim
        half_dim = emb_dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = x[:, None]
        freqs = x * self.weights[None, :] * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        if self.emb_dim % 2 == 1:
            fouriered = torch.nn.functional.pad(fouriered, (0, 1, 0, 0))
        return fouriered


class TimeEmbbeding(nn.Module):
    def __init__(
            self, 
            emb_dim = 64,  
            pos_embedder = SinusoidalPosEmb,
            pos_embedder_kwargs = {},
            act_name=("SWISH", {}) # Swish = SiLU 
        ):
        super().__init__()
        self.emb_dim = emb_dim
        self.pos_emb_dim =  pos_embedder_kwargs.get('emb_dim', emb_dim//4)
        pos_embedder_kwargs['emb_dim'] = self.pos_emb_dim
        self.pos_embedder = pos_embedder(**pos_embedder_kwargs)
        

        self.time_emb = nn.Sequential(
            self.pos_embedder,
            nn.Linear(self.pos_emb_dim, self.emb_dim),
            get_act_layer(act_name),
            nn.Linear(self.emb_dim, self.emb_dim)
        )
    
    def forward(self, time):
        return self.time_emb(time)


class DiagonalGaussianDistribution(nn.Module):
    def forward(self, x):
        mean, logvar = torch.chunk(x, 2, dim=1)
        logvar = torch.clamp(logvar, -30.0, 20.0)
        std = torch.exp(0.5 * logvar)
        sample = torch.randn(mean.shape, generator=None, device=x.device)
        z = mean + std * sample

        batch_size = x.shape[0]
        var = torch.exp(logvar)
        kl = 0.5 * torch.sum(torch.pow(mean, 2) + var - 1.0 - logvar) / batch_size

        return z, kl 

  
class Discriminator(nn.Module):
    def __init__(self, 
        in_channels=1, 
        spatial_dims = 3,
        hid_chs =    [32,       64,      128,      256,  512],
        kernel_sizes=[(1,3,3), (1,3,3), (1,3,3),    3,   3],
        strides =    [  1,     (1,2,2), (1,2,2),    2,   2],
        act_name=("Swish", {}),
        norm_name = ("GROUP", {'num_groups':32, "affine": True}),
        dropout=None
        ):
        super().__init__()

        self.inc =  BasicBlock(
            spatial_dims=spatial_dims, 
            in_channels=in_channels, 
            out_channels=hid_chs[0],  
            kernel_size=kernel_sizes[0], # 2*pad = kernel-stride -> kernel = 2*pad + stride => 1 = 2*0+1, 3, =2*1+1, 2 = 2*0+2, 4 = 2*1+2 
            stride=strides[0], 
            norm_name=norm_name, 
            act_name=act_name, 
            dropout=dropout,
        )

        self.encoder = nn.Sequential(*[
            BasicBlock(
                spatial_dims=spatial_dims, 
                in_channels=hid_chs[i-1], 
                out_channels=hid_chs[i],  
                kernel_size=kernel_sizes[i], 
                stride=strides[i], 
                act_name=act_name, 
                norm_name=norm_name, 
                dropout=dropout)
            for i in range(1, len(hid_chs))
        ])

 
        self.outc =  BasicBlock( 
            spatial_dims=spatial_dims, 
            in_channels=hid_chs[-1], 
            out_channels=1,  
            kernel_size=3, 
            stride=1, 
            act_name=None, 
            norm_name=None, 
            dropout=None,
            zero_conv=True
        )

    def forward(self, x):
        x = self.inc(x)
        x = self.encoder(x)
        return self.outc(x)

class NLayerDiscriminator(nn.Module):
    def __init__(self, 
        in_channels=1, 
        spatial_dims = 3,
        hid_chs =    [64, 128, 256, 512, 512],
        kernel_sizes=[4,   4,   4,  4,   4],
        strides =    [2,   2,   2,  1,   1],
        act_name=("LeakyReLU", {'negative_slope': 0.2}),
        norm_name = ("BATCH", {}),
        dropout=None
        ):
        super().__init__()

        self.inc =  BasicBlock(
            spatial_dims=spatial_dims, 
            in_channels=in_channels, 
            out_channels=hid_chs[0],  
            kernel_size=kernel_sizes[0], 
            stride=strides[0], 
            norm_name=None, 
            act_name=act_name, 
            dropout=dropout,
        )

        self.encoder = nn.Sequential(*[
            BasicBlock(
                spatial_dims=spatial_dims, 
                in_channels=hid_chs[i-1], 
                out_channels=hid_chs[i],  
                kernel_size=kernel_sizes[i], 
                stride=strides[i], 
                act_name=act_name, 
                norm_name=norm_name, 
                dropout=dropout)
            for i in range(1, len(strides))
        ])


        self.outc =  BasicBlock( 
            spatial_dims=spatial_dims, 
            in_channels=hid_chs[-1], 
            out_channels=1,  
            kernel_size=4, 
            stride=1, 
            norm_name=None, 
            act_name=None, 
            dropout=False,
        )

    def forward(self, x):
        x = self.inc(x)
        x = self.encoder(x)
        return self.outc(x)
    

class ConditionMLP(nn.Module):
    def __init__(self, in_features, out_features, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_features)
        )

    def forward(self, x):
        return self.net(x)


class VAE(BasicModel):
    def __init__(
        self,
        in_channels = 3, 
        out_channels = 3, 
        spatial_dims = 2,
        emb_channels = 4,
        hid_chs = [64, 128, 256, 512],
        kernel_sizes = [3, 3, 3, 3],
        strides = [1, 2, 2, 2],
        norm_name = ("GROUP", {'num_groups':8, "affine": True}),
        act_name = ("Swish", {}),
        time_embedder = None,
        time_embedder_kwargs = {}, 
        dropout = None,
        use_res_block = True,
        deep_supervision = False,
        learnable_interpolation = True,
        use_attention = 'none',
        embedding_loss_weight = 1e-6,
        perceiver = LPIPS, 
        perceiver_kwargs = {},
        perceptual_loss_weight = 1.0,
        optimizer = torch.optim.Adam, 
        optimizer_kwargs = {'lr':1e-4},
        lr_scheduler = None, 
        lr_scheduler_kwargs = {},
        loss = torch.nn.L1Loss,
        loss_kwargs = {'reduction': 'none'},
        use_ssim_loss = True,
        use_perceptual_loss = True,
        sample_every_n_steps = 1000
    ):
        super().__init__(
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            lr_scheduler=lr_scheduler,
            lr_scheduler_kwargs=lr_scheduler_kwargs
        )
        self.sample_every_n_steps=sample_every_n_steps
        self.loss_fct = loss(**loss_kwargs)
        self.use_ssim_loss = use_ssim_loss
        self.use_perceptual_loss = use_perceptual_loss
        self.embedding_loss_weight = embedding_loss_weight
        self.perceiver = perceiver(**perceiver_kwargs).eval() if perceiver is not None else None 
        self.perceptual_loss_weight = perceptual_loss_weight
        use_attention = use_attention if isinstance(use_attention, list) else [use_attention] * len(strides) 
        self.depth = len(strides)
        self.deep_supervision = deep_supervision
        downsample_kernel_sizes = kernel_sizes
        upsample_kernel_sizes = strides 

        self.save_hyperparameters()

        # -------- Time/Position embedding ---------
        if time_embedder is not None:
            self.time_embedder = time_embedder(**time_embedder_kwargs)
            time_emb_dim = self.time_embedder.emb_dim
        else:
            self.time_embedder = time_emb_dim = None 

        # ----------- In-Convolution ------------
        ConvBlock = UnetResBlock if use_res_block else UnetBasicBlock
        self.inc = ConvBlock(
            spatial_dims, 
            in_channels, 
            hid_chs[0], 
            kernel_size=kernel_sizes[0], 
            stride=strides[0],
            act_name=act_name, 
            norm_name=norm_name,
            emb_channels=time_emb_dim
        )

        # ----------- Encoder ----------------
        self.encoders = nn.ModuleList([
            DownBlock(
                spatial_dims = spatial_dims, 
                in_channels = hid_chs[i-1], 
                out_channels = hid_chs[i], 
                kernel_size = kernel_sizes[i], 
                stride = strides[i],
                downsample_kernel_size = downsample_kernel_sizes[i],
                norm_name = norm_name,
                act_name = act_name,
                dropout = dropout,
                use_res_block = use_res_block,
                learnable_interpolation = learnable_interpolation,
                use_attention = use_attention[i],
                emb_channels = time_emb_dim
            )
            for i in range(1, self.depth)
        ])

        # ----------- Out-Encoder ------------
        self.out_enc = nn.Sequential(
            BasicBlock(spatial_dims, hid_chs[-1], 2*emb_channels, 3),
            BasicBlock(spatial_dims, 2*emb_channels, 2*emb_channels, 1)
        )


        # ----------- Reparameterization --------------
        self.quantizer = DiagonalGaussianDistribution()    


        # ----------- In-Decoder ------------
        self.inc_dec = ConvBlock(spatial_dims, emb_channels, hid_chs[-1], 3, act_name=act_name, norm_name=norm_name, emb_channels=time_emb_dim) 

        # ------------ Decoder ----------
        self.decoders = nn.ModuleList([
            UpBlock(
                spatial_dims = spatial_dims, 
                in_channels = hid_chs[i+1], 
                out_channels = hid_chs[i],
                kernel_size=kernel_sizes[i+1], 
                stride=strides[i+1], 
                upsample_kernel_size=upsample_kernel_sizes[i+1],
                norm_name=norm_name,  
                act_name=act_name, 
                dropout=dropout,
                use_res_block=use_res_block,
                learnable_interpolation=learnable_interpolation,
                use_attention=use_attention[i],
                emb_channels=time_emb_dim,
                skip_channels=0
            )
            for i in range(self.depth - 1)
        ])

        # --------------- Out-Convolution ----------------
        self.outc = BasicBlock(spatial_dims, hid_chs[0], out_channels, 1, zero_conv=True)
        
        # if isinstance(deep_supervision, bool):
        deep_supervision = self.depth - 1 if deep_supervision else 0 
            
        self.outc_ver = nn.ModuleList([
            BasicBlock(spatial_dims, hid_chs[i], out_channels, 1, zero_conv=True) 
            for i in range(1, deep_supervision + 1)
        ])

        self.save_hyperparameters()
        

    def encode_timestep(self, timestep):
        if self.time_embedder is None:
            return None 
        else:
            return self.time_embedder(timestep)
    
    def encode(self, x, emb=None):
        h = self.inc(x, emb=emb)
        for i in range(len(self.encoders)):
            h = self.encoders[i](h, emb=emb)
        z = self.out_enc(h)
        z, _ = self.quantizer(z)
        return z 
            
    def decode(self, z, emb=None):
        h = self.inc_dec(z, emb=emb)
        for i in range(len(self.decoders), 0, -1):
            h = self.decoders[i-1](h, emb=emb)
        x = self.outc(h)
        return x 

    def forward(self, x_in, timestep=None):
        # --------- Time Embedding -----------
        emb = self.encode_timestep(timestep)

        # --------- Encoder --------------
        h = self.inc(x_in, emb=emb)
        for i in range(len(self.encoders)):
            h = self.encoders[i](h, emb=emb)
        z = self.out_enc(h)

        # --------- Quantizer --------------
        z_q, emb_loss = self.quantizer(z)

        # -------- Decoder -----------
        out_hor = []
        h = self.inc_dec(z_q, emb=emb)
        for i in range(len(self.decoders) - 1, -1, -1):
            if i < len(self.outc_ver):
                out_hor.append(self.outc_ver[i](h))  
            h = self.decoders[i](h, emb=emb)
        out = self.outc(h)
   
        return out, out_hor[::-1], emb_loss 
    
    def perception_loss(self, pred, target, depth=0):
        if (self.perceiver is not None) and (depth < 2):
            self.perceiver.eval()
            return self.perceiver(pred, target) * self.perceptual_loss_weight
        else:
            return 0 
    
    def ssim_loss(self, pred, target):
        return 1 - ssim(
            ((pred + 1) / 2).clamp(0, 1), 
            (target.type(pred.dtype) + 1) / 2, data_range=1, size_average=False, nonnegative_ssim=True
        ).reshape(-1, *(1,) * (pred.ndim - 1))
    
    def rec_loss(self, pred, pred_vertical, target):
        interpolation_mode = 'nearest-exact'
        
        # compute reconstruction loss
        perceptual_loss = self.perception_loss(pred[:, 0, None], target[:, 0, None]) if self.use_perceptual_loss else 0
        ssim_loss = self.ssim_loss(pred, target) if self.use_ssim_loss else 0
        pixel_loss = self.loss_fct(pred, target)

        loss = torch.mean(perceptual_loss + ssim_loss + pixel_loss)
        
        # Note this is include in Stable-Diffusion but logvar is not used in optimizer 
        # rec_loss = rec_loss/ torch.exp(self.logvar) + self.logvar 
        
        # loss += torch.sum(rec_loss) / pred.shape[0]  

        for i, pred_i in enumerate(pred_vertical): 
            target_i = F.interpolate(target, size=pred_i.shape[2:], mode=interpolation_mode, align_corners=None)  
            rec_loss_i  = self.loss_fct(pred_i, target_i) + self.perception_loss(pred_i, target_i) + self.ssim_loss(pred_i, target_i)
            # rec_loss_i = rec_loss_i/ torch.exp(self.logvar_ver[i]) + self.logvar_ver[i] 
            loss += torch.sum(rec_loss_i) / pred.shape[0]  

        return loss
    
    def _step(self, batch, batch_idx, split, step):
        # ------------------------- Get Source/Target ---------------------------
        # x, t = batch
        x, = batch
        target = x
        
        if self.time_embedder is None:
            t = None
        
        # ------------------------- Run Model ---------------------------
        pred, pred_vertical, emb_loss = self(x, timestep=t)

        # ------------------------- Compute Loss ---------------------------
        loss = self.rec_loss(pred, pred_vertical, target)
        loss += emb_loss * self.embedding_loss_weight
         
        # --------------------- Compute Metrics  -------------------------------
        with torch.no_grad():
            logging_dict = {'loss': loss, 'emb_loss': emb_loss}
            logging_dict['L2'] = torch.nn.functional.mse_loss(pred, target)
            logging_dict['L1'] = torch.nn.functional.l1_loss(pred, target)
            logging_dict['mask_rec_loss'] = torch.sum(self.loss_fct(pred, target)) / pred.shape[0]      
            logging_dict['ssim'] = ssim((pred + 1) / 2, (target.type(pred.dtype) + 1) / 2, data_range=1)

        # ----------------- Log Scalars ----------------------
        for metric_name, metric_val in logging_dict.items():
            self.log('{}/{}'.format(split, metric_name), metric_val, prog_bar=True,
                on_step=True, on_epoch=True, sync_dist=True, logger=True
            )

        # -------------------------- Logging ---------------------------
        with torch.no_grad():
            if split == 'val' and (step + 1) % 100 == 0: 
                self.log_images(x, pred, label = 'original - reconstructed')
    
        return loss
    
    def log_images(self, *args, **kwargs):
        if self.trainer.global_rank == 0:
            # at this point x and x_hat are of shape [B, C, 128, 128]
            spatial_stack = lambda x: torch.cat([
                torch.hstack([img for img in x[:, idx, ...]]) for idx in range(x.shape[1])
            ], dim=0)

            args = [spatial_stack(arg) for arg in args]
            
            img = torch.cat(args, dim=0)

            # [-1, 1] => [0, 255]
            img = img.add(1).div(2).mul(255).clamp(0, 255).to(torch.uint8)
            
            wandb.log({
                'Reconstruction examples': wandb.Image(
                    img.detach().cpu().numpy(), 
                    caption='{} ({})'.format(self.trainer.global_step, kwargs['label'])
                )
            })