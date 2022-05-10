
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/UCB-stat-159-s22/hw07-Group14/HEAD?labpath=main.ipynb)

## Single Cell Sequencing Analysis Project

Jennefer, Claudea; Kim, Wendy; Tsai, Gordon; Villouta, Catalina

## Project Scope

In this project we work directly with public available single cell RNA-seq data with the aim of classifying Mus musculus (a mouse) cells to the appropiate organ they came from. Due to computational resources we only worked with cells from kidney and liver, however the work presented here is generalizable to as many organs as needed.

## Project Goals

- Represent the cells in a lower-dimensional space.
- Implement an autoencoder to find a latent representation of the data.
- Compare two-dimensional representations of our data using PCA and the latent space defined by the autoencoder + t-SNE.
- Use the trained encoder to classify unlabeled cells as one of the cell types labeled in the original data (kidney or liver cells).
- Show the performance of the classifier.

## Dependencies
All dependencies are listed in `environment.yml` and `book-requirements.txt`

## Dataset
This project's relevant datasets span cell information from kidney and liver organs. We obtained them from the Tabula Muri's project, a compendium of single-cell transcriptome data from the  Mus musculus organism (specific links are shown in the references). The data used in this project can be found in the data folder:
- `Kidney-counts.csv`
- `Liver-counts.csv`

## Setup
- Create Virtual Environment:
	- using `environment.yml` in terminal
		- `mamba env create -f environment.yml -p ~/envs/genes`
		- `python -m ipykernel install --user --name genes --display-name "IPython - genes"`
	- using Makefile in terminal: `make env`
	
- Activate Virtual Environment:
	- `conda activate genes`

- Install genetools package from source, via:
	- `pip install .`
	
- Run analysis using Makefile in terminal: 
	- `make all`

## Run tests
- Use the command `pytest genetools` on the terminal to run all the tests.

## Reference
We obtained the data from the Tabula Muris project released in 2017 by The Chan Zuckerberg Biohub. All matrices of gene-cell counts and metadata are available as CSVs on [Figshare](https://figshare.com/articles/dataset/Single-cell_RNA-seq_data_from_Smart-seq2_sequencing_of_FACS_sorted_cells_v2_/5829687?file=10700143). We specifically used the data for kidney and liver cells from the FACS-based full-length transcript analysis released in 2018. 

- Consortium, Tabula Muris; Webber, James; Batson, Joshua; Pisco, Angela (2018): Single-cell RNA-seq data from Smart-seq2 sequencing of FACS sorted cells (v2). figshare. Dataset. 
[![DOI](https://figshare.com/badge/DOI/10.6084/m9.figshare.5829687.v8.svg)](https://doi.org/10.6084/m9.figshare.5829687.v8)
