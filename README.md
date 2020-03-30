# <p align="center">CEREBRUM</p>
### <p align="center">CEREBRUM: a fast and fully-volumetric Convolutional Encoder-decodeR for weakly supervised sEgmentation of BRain strUctures from out-of-the-scanner MRI</p>

<p align="center"><img src="https://github.com/denbonte/CER3BRuM/blob/master/assets/logo_large.png" alt="CEREBRuM logo" width="150"></p>

## Introduction
This GitHub repository contains the source code for "[CEREBRUM: a fast and fully-volumetric Convolutional Encoder-decodeR for weakly supervised sEgmentation of BRain strUctures from out-of-the-scanner MRI](https://www.sciencedirect.com/science/article/pii/S1361841520300542)".

Unlike the other models found in the relevant literature, CEREBRuM approaches the segmentation problem without partitioning the testing volume. This allows the model to capture spatial relations between structures, such as their absolute and relative positioning within the brain, and use such features to provide precise and consistent results.

## Getting Started

### Install Dependencies
In order to run the code found in this repository, all the dependencies found in `requirements.txt` must be satisfied. Once the repo is downloaded, just run:

```
pip install -r requirements.txt
```

to install all the dependencies.<br>
<i><b>N.B.</b> `tensorflow-gpu` is not needed neither for the training nor for the testing, but performance greatly degrades if the use of GPUs is not supported.</i>


## Want to Read More?
We will be releasing a blogpost soon: stay tuned!

If you find this code useful in your research, please consider citing the aforementioned following the `bibtex` string below:

```
@article{bontempi2020cerebrum,
  title={CEREBRUM: a fast and fully-volumetric Convolutional Encoder-decodeR for weakly-supervised sEgmentation of BRain strUctures from out-of-the-scanner MRI},
  author={Bontempi, Dennis and Benini, Sergio and Signoroni, Alberto and Svanera, Michele and Muckli, Lars},
  journal={Medical Image Analysis},
  pages={101688},
  year={2020},
  publisher={Elsevier}
}
```
