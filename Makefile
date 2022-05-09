env:
<<<<<<< HEAD
	mamba env create -f genes.yml -p ~/envs/genes
=======
	mamba env create -f environment.yml -p ~/envs/genes
>>>>>>> 5017340b2fea7e7fe457220a19251a6875500407
	bash -ic 'conda activate genes;python -m ipykernel install --user --name genes --display-name "IPython - genes"'

all:
	jupyter execute Single-cell-analysis.ipynb