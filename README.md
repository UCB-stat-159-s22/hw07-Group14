## reproducibility of Genes


[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/UCB-stat-159-s22/hw07-Group14.git/HEAD?labpath=Single-cell-analysis.ipynb)

**Note:** This repository is public so that Binder can find it. All code and data is based on the original [LIGO Center for Open Science Tutorial Repository](https://github.com/losc-tutorial/LOSC_Event_tutorial). This repository is a class exercise that restructures the original LIGO code for improved reproducibility, as a homework assignment for the [Spring 2022 installment of UC Berkeley's Stat 159/259 course, _Reproducible and Collaborative Data Science_](https://ucb-stat-159-s22.github.io). Authorship of the original analysis code rests with the LIGO collaboration.


# Single Cell Sequencing Analysis Project

Jennefer, Claudea; Kim, Wendy; Tsai, Gordon; Villouta, Catalina

## Project Scope
.....


## Dependencies
All dependencies are listed in `genes.yml`

## Dataset
.....

## Setup
- Create Virtual Environment: using `genes.yml` in terminal
	- `mamba env create -f genes.yml -p ~/envs/genes`
	- `python -m ipykernel install --user --name genes --display-name "IPython - genes"`
	
- Activate Virtual Environment:
	conda activate genes
	
- Run analysis using Makefile in terminal:
...


## Reference
....