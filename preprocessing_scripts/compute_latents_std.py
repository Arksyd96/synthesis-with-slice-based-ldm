import argparse
import torch
from tqdm import tqdm
import sys

from modules.models.autoencoders import VAEGAN, VAE
from modules.data_preprocessing import BRATSDataModule

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--model-path', type=str, required=True)
    parser.add_argument('-c', '--ae-class', type=str, required=True, choices=['VAEGAN', 'VAE'], help='Class of the associated latent autoencoder')
    parser.add_argument('-w', '--write', action='store_true', default=False)
    parser.add_argument('-d', '--file-dir', type=str, default='.', required=False)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    autoencoder = getattr(sys.modules[__name__], args.ae_class)
    model = autoencoder.load_from_checkpoint(args.model_path).to(device)
    model.eval()

    datamodule = BRATSDataModule(
        data_dir        = './data/second_stage_dataset_192x192.npy',
        train_ratio     = 1.0,
        norm            = 'centered-norm', 
        batch_size      = 2,
        num_workers     = 12,
        dtype           = torch.float32,
        include_radiomics = False
    )

    datamodule.prepare_data()
    datamodule.setup()

    latents = []
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(datamodule.train_dataloader(), position=0, leave=True, desc='Encoding')):
            x = batch[0].to(device)
            x_hat = model.encode(x, emb=None)
            latents.append(x_hat.detach())
        
        latents = torch.cat(latents, dim=0)
        print(latents.shape)
        std = latents.std()

    print('std: {}'.format(std))

    if args.write:
        with open('{}/std.txt'.format(args.file_dir), 'w') as f:
            f.write('std: {}'.format(std.cpu().numpy()))
