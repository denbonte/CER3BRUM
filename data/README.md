# Data Directory `data/`


## Directory Structure

In our work, we used out-of-the-scanner data. The only pre-process the data went through was the conversion from DICOM to the NIfTI format, carried out using [dcm2niix](https://github.com/rordenlab/dcm2niix).

For the following reason, in order to run the code found in this repo "off-the-shelf", data must stored in NIfTI files (`.nii` or `.nii.gz`). We used [Nipy's NiBabel](https://nipy.org/nibabel/) to handle such MR Images.

Applying changes in these regards (e.g., to use other file formats, or handle data exploiting other python libraries) it's pretty straight forward, as the aforementioned library is used for loading and saving operations.

The data in this folder should be organised as follows:

```
data/
    |_ my_dataset/
                 |_ training/
                 |          |_ $TRVOL1_ID/
                 |          |            |_ ${TRVOL1_ANAT_ID}.nii.gz
                 |          |            |_ ${TRVOL1_SEGM_ID}.nii.gz
                 |          |_ $TRVOL2_ID/
                 |          |            |_ ${TRVOL2_ANAT_ID}.nii.gz
                 |          |            |_ ${TRVOL2_SEGM_ID}.nii.gz
                 ...        ...
                 |
                 |_ testing
                 |          |_ $TSTVOL1_ID/
                 |          |             |_ ${TSTVOL1_ANAT_ID}.nii.gz
                 |          |             |_ ${TTSTVOL1_SEGM_ID}.nii.gz
                 |          |_ $TSTVOL2_ID/
                 |          |             |_ ${TSTVOL2_ANAT_ID}.nii.gz
                 |          |             |_ ${TSTVOL2_SEGM_ID}.nii.gz
                 ...        ...
                 |
                 |_ validation
                 |          |_ $VALVOL1_ID/
                 |          |             |_ ${VALVOL1_ANAT_ID}.nii.gz
                 |          |             |_ ${VALVOL1_SEGM_ID}.nii.gz
                 |          |_ $VALVOL2_ID/
                 |          |             |_ ${VALVOL2_ANAT_ID}.nii.gz
                 |          |             |_ ${VALVOL2_SEGM_ID}.nii.gz
                 ...        ...
```

`TRVOL*_ID`, together with `TRVOL*_ANAT_ID` and `TRVOL*_ANAT_ID` are found according to user-set variables in the source code. In particular, every subject  (folder) should be identified by one (unique) ID, and the name of the NIfTI files should be derived directly from the latter. For instance:

```
anat_suffix = '_anat_1mm.nii.gz'
segm_suffix = '_segm_1mm.nii.gz'


data/
    |_ my_lab_3T_data/
                     |_ training/
                     |          |_ $ABCD00/
                     |          |         |_ ABC00_anat_1mm.nii.gz
                     |          |         |_ ABC00_segm_1mm.nii.gz
                     |          |_ $ABCD01/
                     |          |         |_ ABC01_anat_1mm.nii.gz
                     |          |         |_ ABC01_segm_1mm.nii.gz
                     ...        ...
```

For further information about the aforementioned user-set variables, see `../src/README.md`,

## Publicly Available Test Data

Part of the data we used for testing our model are [made available through OpenNeuro](https://openneuro.org/datasets/ds002207/versions/1.0.0). These are the same volumes that were used for the experts' qualitative assessment (built exploiting [PsychoPy](https://www.psychopy.org), and available under the directory `experts_survey` of this very repository).

Along with the publicly available data, an 8-class segmentation mask is provided. The segmented classes (and the color code used in the notebooks and in the paper) are:

| Class ID | Substructure/Tissue |    Color    |
|:--------:|:-------------------:|:-----------:|
|     0    |      Background     | Transparent |
|     1    |     Grey matter     | Light green |
|     2    |    Basal ganglia    |  Dark green |
|     3    |     White matter    |     Red     |
|     4    |         CSF         |  Light blue |
|     5    |      Ventricles     |     Blue    |
|     6    |      Cerebellum     |    Yellow   |
|     7    |      Brainstem      |     Pink    |


Such ground truth was obtained starting from FreeSurfer's `recon-all` procedure, merging the classes such that only the ones [used in the MICCAI MRBrainS13 and MRBrainS18 challenges](https://mrbrains13.isi.uu.nl/data/) were kept (except the less numerous classes - "White matter lesions", "Infarction" and "Other" - not directly obtainable from the atlas-based segmentation, or for which we considered the latter to be too little reliable).