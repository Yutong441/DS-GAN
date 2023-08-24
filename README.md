# Quantitative comparisons of GAN-synthesized DWI images
## Installation
```bash
# requirement: conda and apptainer installed
make VENV=dwi
```

## Training
```bash
python test/train.py --data_root=SCANS_t0
# the data root is the ID supplied in the assemble_data.py script combined with
# the modality of prediction.
```
