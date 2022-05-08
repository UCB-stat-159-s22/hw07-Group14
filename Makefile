env:
	mamba env create -f genes.yml -p ~/envs/genes
	bash -ic 'conda activate genes;python -m ipykernel install --user --name genes --display-name "IPython - genes"'

all:
	jupyter execute Single-cell-analysis.ipynb