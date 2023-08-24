# Build a conda and a pip environment
# conda environment: has pytorch and mrtrix (which can only be installed in
# conda)
# pip environment: has tensorflow1.5 (need a separate environment due to being
# dependent on different versions of cuda and cudnn)
SHELL := /bin/bash
VENV = venv_dwi
VENV_DIR = "."
PYTHON1 = 3.8
PYTHON2 = 3.7
PIP1 = $(HOME)/.conda/envs/$(VENV)/bin/pip
PIP2 = $(VENV_DIR)/$(VENV)/bin/pip

CONDA_ACTIVATE = source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate

all: WMH environment
.PHONY = $(VENV_DIR)/HyperMapp3r/depends/c3d singularity

$(VENV_DIR)/HyperMapp3r/depends/c3d:
	git clone https://github.com/AICONSlab/HyperMapp3r.git $(VENV_DIR)/HyperMapp3r; \
	cd $(VENV_DIR)/HyperMapp3r; \
	/bin/bash install_depends.sh; \
	rm install_log.out

WMH: $(VENV_DIR)/HyperMapp3r/depends/c3d
	cd $(VENV_DIR); \
	python$(PYTHON2) -m venv $(VENV); \
	source $(VENV)/bin/activate; \
	$(PIP2) install --upgrade pip; \
	git clone https://www.github.com/keras-team/keras-contrib.git; \
	cd keras-contrib; \
	$(VENV_DIR)/$(VENV)/bin/python setup.py install; \
	cd ../HyperMapp3r; \
	$(PIP2) install -e .[hypermapper_gpu]; \
	$(PIP2) install protobuf==3.20.0; \
	$(PIP2) install h5py==2.10.0; \
	deactivate

environment: requirements.txt
	conda create --name $(VENV) python=$(PYTHON1); \
	$(CONDA_ACTIVATE) $(VENV); \
	conda install -c mrtrix3 mrtrix3 -y; \
	$(PIP1) install --upgrade pip; \
	$(PIP1) install -r requirements.txt; \
	cd $(VENV_DIR); \
	git clone https://github.com/MIC-DKFZ/HD-BET; \
	cd HD-BET; \
	$(PIP1) install -e . ; \
	conda deactivate

singularity:
	echo "not installing singularity"
	# cd $(VENV_DIR); \
	apptainer pull docker://leonyichencai/synb0-disco:v3.0
