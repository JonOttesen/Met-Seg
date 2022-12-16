# Met Seg

The GitHub repo for *2.5D and 3D Segmentation of Brain Metastases with Deep Learning on Multinational MRI Data*. This repository provides the source code for the models (HRNetV2), model weight, training schematic, evaluation metrics, and a simple inference script.

## Dataset

The models were trained on the "BrainMetShare" dataset available [here](https://aimi.stanford.edu/brainmetshare). The OUH dataset used for evaluation will be made publically available later, and access is granted upon request.

## Inference

### Prerequisites

For a given study, all images must be co-registered and it is recommended that they follow the "LPS" direction (this shouldn't be too important). Second, all images should either be brain extracted or have a corresponding brain mask named ```*_mask.nii.gz```.

### Model Weights

The model weights can be downloaded from:
| Type   | Link |
|--------|------|
| 2D     |      |
| 3D     |      |
| nnUNet |      |

### General Information

All models were trained on four sequences with the following input order: BRAVO, T1 post gd, T1 pre gd, and FLAIR. Note, the 2D and 3D variants are trained to handle missing input sequences. In those cases, just have the channels corresponding to the missing sequences be zeroes and multiply the input with:
$$
\begin{align}
inp = inp\cdot\frac{1}{1-p},
\end{align}
$$
where p is the fraction of included sequences (p=0.75 when 3 sequences are included, this happens automatically).

### File structure

To run the script, the file structure must follow the following order:
- Mets
    - Scan 1
        - bravo.nii
        - t1_pre.nii
        - t1_post.nii
        - flair.nii
    - Scan 2
        - bravo.nii
        - t1_pre.nii
        - t1_post.nii
        - flair.nii

**Note, the naming convention is important and they must be in nifti format!!!**

BRAVO sequence (MPRAGE) - "bravo"\
T1 post-contrast sequence (CUBE/SPACE) - "t1_post"\
T1 pre-contrast sequence (CUBE/SPACE) - "t1_pre"\
FLAIR sequence (CUBE/SPACE) - "flair"


### Predict

To use either the 3D network call the ```predict.py``` script as follows:
```
predict-met -i where/files/are/stored -i where/files/are/saved -c checkpoints/ --device cuda:0
```
where ```-i``` is the location of the nifti files, ```-o``` where the files are stored (optional, by default they are stored in the same directory as the input), ```-c``` where the weights are stored, and ```--device``` which device to use for the prediction (cuda or cpu). 2D support has not been added as the 3D variant seemed to have the most promise.

### Good to know

It is recommended to have a GPU, but not recommended. In total, the inference takes about ... on a Nvidia RTX 3060 laptop version, and ... on an AMD Ryzen 9 5900HS. The model takes about 2.5GB of GPU memory.

### Installation

The dependencies can be downloaded by running:
```
cd met-seg
conda env create -f environment.yml
```
After the environment/packages have been installed. The program ```predict-met``` can be installed so you can run the prediction without being in this repository. Note, this only works on Linux. Run the following command in the *met-seg* repo:
```
pip install -e .
```