## ----------------------------------------
## Compute the voxelwise mean and standard
## deviation across all the training volumes
## ----------------------------------------
## 
## ----------------------------------------
## Author: Dennis Bontempi, Michele Svanera
## Version: 2.0
## Email: dennis.bontempi@maastro.nl
## Modified: 30 MAR 19
## ----------------------------------------


import os
import sys
import numpy as np
import nibabel as nib
import argparse
import time


## ----------------------------------------
##         USER-SPECIFIED VARIABLES
## ----------------------------------------

# anatomical volume suffix (to append to every volume name) - see data/README.md for additional info (str)
anat_suffix = '_anatomical_3T_Glasgow_1mm_Crop_1.nii.gz'


## ----------------------------------------
##             ARGUMENT PARSING
## ----------------------------------------

parser  = argparse.ArgumentParser(description='Compute the voxel-wise mean and standard deviation across a set of (anatomical MRI) volumes.')

parser.add_argument('--norm', 
                    help    = 'Specify whether or not normalise the volumes between 0 and 1 (for comparison purposes). Default is "False".', 
                    choices = ['True', 'False'],
                    default = 'False',
                    )


required = parser.add_argument_group('Required arguments')

required.add_argument('--dataset_name',
                      help      = 'Name of the dataset (under /data/).',
                      type      = str,
                      required  = True,
                      )

                    
args = parser.parse_args()


apply_norm = True if args.norm == 'True' else 'False'
dataset_name = args.dataset_name


## ----------------------------------------
##                   INIT
## ----------------------------------------


# base path of the folder containing all the volumes (in subfolders or not)
data_path = os.path.join('../data', dataset_name, 'training')

# where to store the output volumes
out_path = os.path.join('../output/zscoring', dataset_name)

if not os.path.exists(out_path):
    os.makedirs(out_path)


# get the list of all the subdirs in data_path
subj_list = sorted(os.listdir(data_path))

# get the dimension from the first subject
data_dims = nib.load(os.path.join(data_path, subj_list[0], subj_list[0] + anat_suffix)).get_data().shape

# store the header (later use)
data_header = nib.load(os.path.join(data_path, subj_list[0], subj_list[0] + anat_suffix)).header

# allocate useful structures
voxelwise_mean = np.zeros((data_dims))
voxelwise_std  = np.zeros((data_dims))
voxelwise_var  = np.zeros((data_dims))

if apply_norm:
    norm_voxelwise_mean = np.zeros((data_dims))
    norm_voxelwise_std  = np.zeros((data_dims))
    norm_voxelwise_var  = np.zeros((data_dims))
    voxelwise_max  = np.zeros((data_dims))
    voxelwise_min  = 1000 * np.ones((data_dims))

    # compute the voxelwise mean 
    print("Computing the voxel-wise maximum and mimimum...")
    sys.stdout.flush()

    # keep track of the time
    start = time.time()

    for subj_num, subj_id in enumerate(subj_list):

        # print progress info
        print('Subject %s (%3d/%3d) \r'%(subj_id, subj_num, len(subj_list))),
        sys.stdout.flush()

        # load the current volume
        mri_path    = os.path.join(data_path, subj_id, subj_id + anat_suffix)
        mri_vol     = nib.load(mri_path).get_data().astype(dtype=np.float32)

        # handle exceptions on data_dims
        if mri_vol.shape != data_dims:
            print('Warning: volume size mismatch (subject %s). Moving on to the next subject...'%(subj_id))
            continue;

        # update the voxelwise max and min
        voxelwise_max = np.maximum(voxelwise_max, mri_vol).astype(dtype=np.float32)
        voxelwise_min = np.minimum(voxelwise_min, mri_vol).astype(dtype=np.float32)

    end             = time.time()
    time_elapsed    = end - start

    print("\nDone (in %2.2f seconds)!"%(time_elapsed))


# compute the voxelwise mean 
print("Computing the voxel-wise mean...")
sys.stdout.flush()

# keep track of the time
start = time.time()

for subj_num, subj_id in enumerate(subj_list):

    # print progress info
    print('Subject %s (%3d/%3d) \r'%(subj_id, subj_num, len(subj_list))),
    sys.stdout.flush()

    # load the current volume
    mri_path    = os.path.join(data_path, subj_id, subj_id + anat_suffix)
    mri_vol     = nib.load(mri_path).get_data().astype(dtype=np.float32)
    
    # handle exceptions on data_dims
    if mri_vol.shape != data_dims:
        print('Warning: volume size mismatch (subject %s). Moving on to the next subject...'%(subj_id))
        continue;

    voxelwise_mean += mri_vol/len(subj_list)

    if apply_norm:

        norm_mri_vol = np.copy(mri_vol)
        norm_mri_vol -= np.min(norm_mri_vol)
        norm_mri_vol /= np.ptp(norm_mri_vol)

        norm_voxelwise_mean += norm_mri_vol/float(len(subj_list))

end             = time.time()
time_elapsed    = end - start

print("\nDone (in %2.2f seconds)!"%(time_elapsed))
    
# save the normalised voxel-wise mean
if apply_norm:

    # save the volume containing the normalised voxel-wise mean
    print('Saving the normalised voxel-wise mean volume at %s...'%(out_path)),
    sys.stdout.flush()
    nifti_to_save = nib.Nifti1Image(norm_voxelwise_mean, affine = data_header.get_sform(), header = data_header)
    nib.save(nifti_to_save, os.path.join(out_path, 'norm_voxelwise_mean.nii.gz'))
    print("Done.")

# save the volume containing the voxel-wise mean
print('Saving the voxel-wise mean volume at %s...'%(out_path)),
sys.stdout.flush()
nifti_to_save = nib.Nifti1Image(voxelwise_mean, affine = data_header.get_sform(), header = data_header)
nib.save(nifti_to_save, os.path.join(out_path, 'voxelwise_mean.nii.gz'))
print("Done.")

# at this point, the variable average_subject subject contains the actual saved volume
# for this reason, don't load it from the .nii.gz but use the variable instead
# compute the voxelwise mean 
print("Computing the voxel-wise standard deviation...")
sys.stdout.flush()

# keep track of the time
start = time.time()


for subj_num, subj_id in enumerate(subj_list):

    # print progress info
    print('Subject %s (%3d/%3d) \r'%(subj_id, subj_num, len(subj_list))),
    sys.stdout.flush()

    # load the current volume
    mri_path    = os.path.join(data_path, subj_id, subj_id + anat_suffix)
    mri_vol     = nib.load(mri_path).get_data().astype(dtype=np.float32)

    # handle exceptions on data_dims
    if mri_vol.shape != data_dims:
        print('Warning: volume size mismatch (subject %s). Moving on to the next subject...'%(subj_id))
        continue;

    voxelwise_var += np.square(mri_vol - voxelwise_mean)/float(len(subj_list))

    if apply_norm:
        norm_mri_vol = np.copy(mri_vol)
        norm_mri_vol -= np.min(norm_mri_vol)
        norm_mri_vol /= np.ptp(norm_mri_vol) 

        norm_voxelwise_var += np.square(norm_mri_vol - norm_voxelwise_mean)/float(len(subj_list))

voxelwise_std = np.sqrt(voxelwise_var)

end             = time.time()
time_elapsed    = end - start

print("\nDone (in %2.2f seconds)!"%(time_elapsed))


# save the normalised voxel-wise mean
if apply_norm:

    norm_voxelwise_std = np.sqrt(norm_voxelwise_var)
 
    # save the volume containing the normalised voxel-wise mean
    print('Saving the normalised voxel-wise standard deviation volume at %s...'%(out_path)),
    sys.stdout.flush()
    nifti_to_save = nib.Nifti1Image(norm_voxelwise_std, affine = data_header.get_sform(), header = data_header)
    nib.save(nifti_to_save, os.path.join(out_path, 'norm_voxelwise_std.nii.gz'))
    print("Done.")

# save the volume containing the voxel-wise standard deviation
print('Saving the voxel-wise standard deviation volume at %s...'%(out_path)),
sys.stdout.flush()
nifti_to_save = nib.Nifti1Image(voxelwise_std, affine = data_header.get_sform(), header = data_header)
nib.save(nifti_to_save, os.path.join(out_path, 'voxelwise_std.nii.gz'))
print("Done.")

    
