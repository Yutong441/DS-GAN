# Quantitative comparisons of GAN-synthesized DWI images
## Installation
```bash
# requirement: conda and apptainer installed
make VENV=dwi
```

Download the model weights from [here](https://drive.google.com/drive/folders/1WidTHCgiDSRceKrWUyrpSTPEPnr0hkaj?usp=sharing).
After you click on the link, you will be prompted to enter a message. 
We will share the weights upon reasonable requests.
Then, place the "results" folder into the main folder (where this README.md is located).

## Training
```bash
python test/train.py --data_root=SCANS_t0
# the data root is the ID supplied in the assemble_data.py script combined with
# the modality of prediction.
```

## Run model
The folder structure should be:

```
Subject1
-------original
---------------T1_MNI.nii.gz
---------------FLAIR_MNI.nii.gz
-------synthetic
Subject2
-------original
---------------T1_MNI.nii.gz
---------------FLAIR_MNI.nii.gz
-------synthetic
```

"T1_MNI.nii.gz" refers to skullstripped, bias-corrected 3D T1 images registered linearly to the MNI space (please use the template in the `results/MNI.nii.gz`).
Please refer to our [original manuscript](https://www.ahajournals.org/doi/full/10.1161/STROKEAHA.124.047449) for preprocessing details.
FLAIR input is not necessary, but if present, would increase the synthesis accuracy.

The command to run the model is:
```bash
python3 run.py --gpu_ids=0 --in_chan=T1,FLAIR --dir_txt=<directory containing image files>
```

To run the model on CPU, set `--gpu_ids=-1`.
If there is only T1 data available, set `--chain=T1`.
