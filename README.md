## Reproducibility of Genes


[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/UCB-stat-159-s22/hw07-Group14/HEAD?labpath=main.ipynb)



# Single Cell Sequencing Analysis Project

Jennefer, Claudea; Kim, Wendy; Tsai, Gordon; Villouta, Catalina

## Project Scope
This project applies PCA and t-SNE to techniques to classify cells from the the aorta, kidney, liver, and lung organs. We can observe whether certain generic attributes of these cells make classification less or more accurate. 


## Dependencies
All dependencies are listed in `environment.yml` and `book-requirements.txt`

## Dataset
The relevant datasets of this project span across cell information from the aorta, kidney, liver, and lung organs.

## Setup
- Create Virtual Environment
	- using `environment.yml` in terminal
		- `mamba env create -f environment.yml -p ~/envs/genes`
		- `python -m ipykernel install --user --name genes --display-name "IPython - genes"`
	- using Makefile in terminal: 'make env'
	
- Activate Virtual Environment:
	conda activate genes
	
- Run analysis using Makefile in terminal: 'make all' 


## Reference
- https://doi.org/10.6084/m9.figshare.5829687.v8
	- FACS.zip
- https://doi.org/10.6084/m9.figshare.5821263.v3
- https://doi.org/10.6084/m9.figshare.5829687.v8
- https://doi.org/10.6084/m9.figshare.5968960.v3
- https://doi.org/10.6084/m9.figshare.5975392.v1
- https://doi.org/10.6084/m9.figshare.5725891.v1
- https://doi.org/10.6084/m9.figshare.5715025.v1https://doi.org/10.6084/m9.figshare.5715040.v1 
