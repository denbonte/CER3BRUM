# Source Directory `src/`

## Directory Structure

The source directory is structured as follows.

```
src/
   |_ CEREBRUM notebook - data.ipynb
   |_ CEREBRUM notebook - testing.ipynb
   |_ CEREBRUM notebook - training.ipynb
   |_ compute_stdz_mat.py
   |_ cer3brum_lib/
```

We provide three interactive Jupyter Notebooks:

- `CEREBRUM notebook - data.ipynb`;
- `CEREBRUM notebook - testing.ipynb`;
- `CEREBRUM notebook - training.ipynb`

in which each step and piece of code is commented extensively. Together with the aforementioned we will also provide (soon!) the python source code, so that the training procedure can be run from command-line.

The `compute_stdz_mat.py` script computes the voxel-wise mean and standard deviation across all the volumes of a specified dataset. Since one of the basic operations performed on the training volumes is the z-scoring, this should be the first step.

<i><b>N.B.</b> NaN raised by voxels with zero variability (e.g. due to zero padding during basic pre-processing of the volume) are taken care of during the training</i>

The `cer3brum_lib` folder contains python scripts useful in the model declaration phase and some other functions.



### Training a New Instance of CEREBRUM


Please, make sure the data is named in the correct way and placed in the right directory structured in the fashion described in `data/README.md`.

An extensive example of the training procedure is provided in the dedicated Jupyter Notebook.

