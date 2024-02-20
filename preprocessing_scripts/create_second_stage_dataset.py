import numpy as np
from nibabel import load
from nibabel.processing import resample_to_output
import os
import argparse
from tqdm import tqdm

def normalize(input_data, norm='centered-norm'):
    assert norm in ['centered-norm', 'z-score', 'min-max'], "Invalid normalization method"

    if norm == 'centered-norm':
        norm = lambda x: (2 * x - x.min() - x.max()) / (x.max() - x.min())
    elif norm == 'z-score':
        norm = lambda x: (x - x.mean()) / x.std()
    elif norm == 'min-max':
        norm = lambda x: (x - x.min()) / (x.max() - x.min())
    return norm(input_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data-path', type=str, required=True, help='Path to the BRATS dataset folder')
    parser.add_argument('-n', '--n-samples', type=int, default=1000, help='Number of patients to load')
    parser.add_argument('-m', '--modalities', type=str, nargs='+', default=['t1', 't1ce', 't2', 'flair', 'seg'], help='Modalities to load')
    parser.add_argument('-t', '--target-shape', type=int, nargs='+', default=[240, 240, 128], help='Target shape to resize the volumes')
    parser.add_argument('-b', '--binarize', action='store_true', help='Binarize the segmentation mask')
    parser.add_argument('-r', '--randomize', action='store_true', help='Randomize the subjects before loading')
    parser.add_argument('-s', '--save-path', type=str, default='./', help='Path to save the npy file')
    args = parser.parse_args()

    print('Loading dataset from NiFTI files... (This script requires a lot of memory)')

    if args.randomize:
        instances = np.random.choice(os.listdir(args.data_path), size=args.n_samples, replace=False)
    else:
        instances = os.listdir(args.data_path)[:args.n_samples]

    data = np.zeros(shape=(
        args.n_samples,
        args.modalities.__len__(),
        args.target_shape[0],
        args.target_shape[1],
        args.target_shape[2]
    ))

    for idx, instance in enumerate(tqdm(instances, position=0, leave=True, desc="Processing patients")):
        # loading models
        volumes = {}
        for _, m in enumerate(args.modalities):
            volumes[m] = load(os.path.join(args.data_path, instance, instance + f'_{m}.nii.gz'))

        # Compute the scaling factors (output will not be exactly the same as defined in OUTPUT_SHAPE)
        orig_shape = volumes[args.modalities[0]].shape
        scale_factor = (orig_shape[0] / args.target_shape[0], # height
                        orig_shape[1] / args.target_shape[1], # width
                        orig_shape[2] / args.target_shape[2]) # depth

        # Resample the image using trilinear interpolation
        # Drop the last extra rows/columns/slices to get the exact desired output size
        for m in args.modalities:
            volumes[m] = resample_to_output(volumes[m], voxel_sizes=scale_factor, order=0).get_fdata()
            volumes[m] = volumes[m][:args.target_shape[0], :args.target_shape[1], :args.target_shape[2]]

        # binarizing the mask (for simplicity)
        if args.binarize and 'seg' in args.modalities:
            volumes['seg'] = (volumes['seg'] > 0).astype(np.float32)

        # stacking, normalizing and transposing ...
        volumes = np.stack([volumes[m] for m in args.modalities], axis=0)

        # patient-wise normalization
        for idx_m in range(args.modalities.__len__()):
            volumes[idx_m, ...] = normalize(volumes[idx_m, ...], 'centered-norm')

        data[idx, ...] = volumes

    print('Final dataset shape: {}'.format(data.shape))

    np.save('{}/second_stage_dataset_{}x{}.npy'.format(
        args.save_path, args.target_shape[0], args.target_shape[1]
    ), data, allow_pickle=True)
    print('Saved at {}'.format(args.save_path))