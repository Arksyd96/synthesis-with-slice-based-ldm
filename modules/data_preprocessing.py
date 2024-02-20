import numpy as np
import torch 
from torch import nn
from torchvision import transforms as T
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split

class IdentityDataset(torch.utils.data.Dataset):
    """
        Simple dataset that returns the same data (d0, d1, ..., dn)
    """
    def __init__(self, *data):
        self.data = data

    def __len__(self):
        return self.data[-1].__len__()

    def __getitem__(self, index):
        return [d[index] for d in self.data]
    

def normalize(input_data, norm='centered-norm'):
    assert norm in ['centered-norm', 'z-score', 'min-max'], "Invalid normalization method"

    if norm == 'centered-norm':
        norm = lambda x: (2 * x - x.min() - x.max()) / (x.max() - x.min())
    elif norm == 'z-score':
        norm = lambda x: (x - x.mean()) / x.std()
    elif norm == 'min-max':
        norm = lambda x: (x - x.min()) / (x.max() - x.min())
    return norm(input_data)


class BRATSDataset(torch.utils.data.Dataset):
    """
        Images always as first argument, labels and other variables as last
        Transforms are applied on first argument only
    """
    def __init__(
        self,
        *data,
        transform       = None,
        resize          = None,
        horizontal_flip = None,
        vertical_flip   = None, 
        random_crop_size = None,
        rotation        = None,
        dtype           = torch.float32
    ):
        super().__init__()
        self.data = data

        if transform is None: 
            self.transform = T.Compose([
                T.Resize(resize) if resize is not None else nn.Identity(),
                T.RandomHorizontalFlip(p=horizontal_flip) if horizontal_flip else nn.Identity(),
                T.RandomVerticalFlip(p=vertical_flip) if vertical_flip else nn.Identity(),
                T.RandomCrop(random_crop_size) if random_crop_size is not None else nn.Identity(),
                T.RandomRotation(rotation, fill=-1) if rotation is not None else nn.Identity(),
                T.ConvertImageDtype(dtype),
                # WARNING: mean and std are not the target values but rather the values to subtract and divide by: [0, 1] -> [0-0.5, 1-0.5]/0.5 -> [-1, 1]
            ])
        else:
            self.transform = transform

    def __len__(self):
        return self.data[-1].__len__()

    def __getitem__(self, index):
        return [self.transform(d[index]) if i == 0 else d[index] for i, d in enumerate(self.data)]
    
    def sample(self, n, transform=None):
        """ sampling randomly n samples from the dataset, apply or not the transform"""
        transform = self.transform if transform is not None else lambda x: x
        idx = np.random.choice(len(self), n)
        return [transform(d[idx]) if i == 0 else d[idx] for i, d in enumerate(self.data)]
    

class BRATSDataModule(LightningDataModule):
    def __init__(
        self,
        train_dir: str, # should target the a npy file generated via create_*.py scripts
        train_ratio: float = 0.8,
        norm = 'min-max',
        batch_size: int = 32,
        num_workers: int = 4,
        shuffle: bool = True,
        horizontal_flip = None,
        vertical_flip = None, 
        rotation = None,
        random_crop_size = None,
        dtype = torch.float32,
        verbose = True,
        include_radiomics = True,
        **kwargs
    ):
        super().__init__()
        self.dataset_kwargs = {
            "horizontal_flip": horizontal_flip,
            "vertical_flip": vertical_flip, 
            "random_crop_size": random_crop_size,
            "rotation": rotation,
            "dtype": dtype
        }

        self.train_dir = train_dir
        self.norm = norm
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.train_ratio = train_ratio
        self.include_radiomics = include_radiomics
        self.verbose = verbose

    def prepare_data(self) -> None:
        pass
        
    def setup(self, stage=None):      
        self.data = np.load(self.train_dir, allow_pickle=True)
        self.data = torch.from_numpy(self.data)

        ##############################
        if self.include_radiomics:
            self.radiomics = np.load('./data/radiomics.npy', allow_pickle=True).tolist()
            self.radiomics = torch.tensor(
                [
                    tuple(self.radiomics[key][i] for key in self.radiomics) 
                    for i in range(len(self.radiomics['w']))
                ],
                dtype=self.dataset_kwargs['dtype']
            )

        # if self.include_radiomics:
            train_images, train_y, val_images, val_y = train_test_split(
                self.data, self.radiomics, train_size=self.train_ratio, random_state=42, shuffle=False
            ) if self.train_ratio < 1 else (self.data, self.radiomics, [], [])

            self.train_dataset = BRATSDataset(train_images, train_y, **self.dataset_kwargs)
            self.val_dataset = BRATSDataset(val_images, val_y)
        
        else:
            train_images, val_images = train_test_split(
                self.data, train_size=self.train_ratio, random_state=42, shuffle=False
            ) if self.train_ratio < 1 else (self.data, [])

            self.train_dataset = BRATSDataset(train_images, **self.dataset_kwargs)
            self.val_dataset = BRATSDataset(val_images)

        log = """
        DataModule setup complete.
        Number of training samples: {}
        Number of validation samples: {}
        Data shape: {}
        Maximum: {}, Minimum: {}
        """.format(
            len(self.train_dataset),
            len(self.val_dataset),
            self.data.shape,
            self.data.max(),
            self.data.min()
        )

        if self.verbose: print(log)
        
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size = self.batch_size,
            num_workers = self.num_workers,
            shuffle = self.shuffle,
            pin_memory = True
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size = self.batch_size,
            num_workers = self.num_workers,
            shuffle = False,
            pin_memory = True
        )
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size = self.batch_size,
            num_workers = self.num_workers,
            shuffle = False,
            pin_memory = True
        )
    


