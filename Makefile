JPG_FILES = $(wildcard figures/*.jpg)
CSV_FILES = $(wildcard tables/*.csv)

#To Build and deploy Jupyter Book locally
.PHONY : html
html: 
	jupyter-book build .
	
.PHONY : html-hub
html-hub: 
	sphinx-build  . _build/html -D html_baseurl=${JUPYTERHUB_SERVICE_PREFIX}/proxy/absolute/8000
	@echo "To see Ligo Book visit: https://stat159.datahub.berkeley.edu/user-redirect/proxy/8000/index.html"
	cd _build/html && python -m http.server
	

# Run Jupyter Notebook to obtain figures, audio files, and table
.PHONY : all
all :
	jupyter execute main.ipynb

# Clean figures, tables, and build book
.PHONY : clean
clean :
	rm -f $(JPG_FILES)
	rm -f $(CSV_FILES)
	rm -rf _build/html/

#Create Environment
.PHONY: env
env:
	mamba env create -f environment.yml -p ~/envs/genes
	bash -ic 'conda activate genes
	python -m ipykernel install --user --name genes --display-name "IPython - genes"'

.PHONY : variables
variables :
	@echo PNG_FILES: $(JPG_FILES)
	@echo CSV_FILES: $(CSV_FILES)