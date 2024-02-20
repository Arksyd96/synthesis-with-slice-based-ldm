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
    parser.add_argument('-t', '--target-shape', type=int, nargs='+', default=[128, 128, 64], help='Target shape to resize the volumes')
    parser.add_argument('-b', '--binarize', action='store_true', help='Binarize the segmentation mask')
    parser.add_argument('-l', '--slice-idx', action='store_true', help='Include or not slice index')
    parser.add_argument('-r', '--remove-empty', action='store_true', help=r'Remove 90% of empty slices (black images)')
    parser.add_argument('-s', '--save-path', type=str, default='./', help='Path to save the npy file')
    args = parser.parse_args()

    # assert seg is last in the list
    assert args.modalities[-1] == 'seg', "The segmentation mask must be the last modality in the list"

    print('Loading dataset from NiFTI files... (This script requires a lot of memory)')

    data = np.zeros(shape=(
        args.n_samples * args.target_shape[2],
        args.modalities.__len__(),
        args.target_shape[0],
        args.target_shape[1],
    ))

    for idx, instance in enumerate(tqdm(os.listdir(args.data_path)[: args.n_samples], position=0, leave=True, desc="Processing patients")):
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

        # saving models in a single array
        volumes = volumes.transpose(3, 0, 1, 2)
        data[idx * args.target_shape[2]: (idx + 1) * args.target_shape[2], ...] = volumes

    if args.slice_idx:
        slice_levels = np.arange(0, args.target_shape[2])[None, :]
        slice_levels = slice_levels.repeat(args.n_samples, axis=0).reshape(-1, 1)

    if args.remove_empty:
        # not working with z-score normalization
        empty_slices_map = data[:, :-1].mean(axis=(1, 2, 3)) <= data.min() + 1e-5 # all modalities except seg
        empty_slices_num = empty_slices_map.sum()
        num_to_set_false = int(empty_slices_num * 0.1)
        print('Removing empty slices (keeping 10% | {} out of {})...'.format(num_to_set_false, empty_slices_num))

        # randomly select 10% of the True values and set them to False (so we remove 90%)
        indices = np.where(empty_slices_map == True)[0]
        indices_to_set_false = np.random.permutation(empty_slices_num)[:num_to_set_false]
        empty_slices_map[indices[indices_to_set_false]] = False

        # removing selected empty slices
        data = data[~empty_slices_map.reshape(-1)]
        if args.slice_idx:
            slice_levels = slice_levels[~empty_slices_map.reshape(-1)]

    assert slice_levels.shape[0] == data.shape[0], "Slice levels and data shape do not match"
    print('Final dataset shape: {}'.format(data.shape))    

    dataset = {}
    dataset['images'] = data
    if args.slice_idx: dataset['slice_idx'] = slice_levels
 
    # saving the dataset as a npy file
    np.save('{}/first_stage_dataset_{}x{}.npy'.format(
        args.save_path, args.target_shape[0], args.target_shape[1], 
    ), dataset, allow_pickle=True)
    print('Saved at {}'.format(args.save_path))