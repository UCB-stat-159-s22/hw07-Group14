
import os
import pytest
import pandas as pd
from genetools import dataloader
from genetools import ml

NUM_GENES = 1000


@pytest.fixture
def data():
	"""
	Returns data as a fixture in conftest, i.e. available to all tests
	and ran only once.
	The first NUM_GENES columns will be features.
	The last column is the target for the supervised learning task.
	After testing the file created is deleted.
	"""
	
	dp = dataloader.DataPreprocessing(
		organs = ["Kidney", "Liver"],
		y_label = "organ",
		input_path_format = "data/{organ}-counts.csv",
		output_path = f"processed_data/data.csv",
		bool_filter_zero_variance = True,
		bool_filter_mean_diff = True,
		bool_take_log = True,
		bool_save = True,
		num_genes = NUM_GENES,
	)
	
	data_run = dp.run()
	
	yield data_run
	
	os.remove(dp.output_path)


@pytest.fixture
def m(data):
	"""
	Returns Models instance with reasonable inputs.
	Notable for testing, max_epochs is kept rather low for efficiency,
	and verbose needs to be kept equal to 0.
	"""
	
	yield ml.Models(
		data=data,
		autoencoder_path="models/autoencoder_best.hdf5",
		encoder_path="models/encoder_best.hdf5",
		classifier_path_format="models/nn_{size}_best.hdf5",
		projection_plot_path="figures/projection.jpg",
		cv_path="tables/cv_results.csv",
		test_performance_path="tables/test_performance.csv",
		roc_path_format="figures/roc_{size}.jpg",
		confusion_matrix_path_format="figures/confusion_matrix_{size}.jpg",
		max_epochs=30,
		verbose=0,
	)


@pytest.fixture
def run_encoder(m):
	"""
	Fits and saves autoencoder and encoder.
	After testing the files created are deleted.
	"""
	
	m.run_encoder()
	
	yield
	
	os.remove(m.autoencoder_path)
	os.remove(m.encoder_path)


@pytest.fixture
def run_2d_plot(run_encoder, m):
	"""
	Creates and saves plot with projection of encoded data and
	alternative PCA approach.
	After testing the file created is deleted.
	"""
	
	m.plot_encoder_2d_and_compare()
	
	yield
	
	os.remove(m.projection_plot_path)


@pytest.fixture
def run_classifier(run_encoder, m):
	"""
	Loads previously trained encoder and trains classifier.
	After testing the files created are deleted.
	"""
	
	m.load_encoder()
	m.run_classifier()
	
	yield
	
	os.remove(m.classifier_path_format.format(size="small"))
	os.remove(m.classifier_path_format.format(size="larger"))


@pytest.fixture
def run_assessment(run_classifier, m):
	"""
	Runs an array of performance assessment including:
	- Cross-validation. Heatmap plots saved.
	- Test set performance. csv file saved.
	- ROC curve. Plot saved.
	After testing the files created are deleted.
	"""
	
	m.run_assessment()
	
	yield
	
	os.remove(m.cv_path)
	os.remove(m.test_performance_path)
	os.remove(m.roc_path_format.format(size="small"))
	os.remove(m.roc_path_format.format(size="larger"))
	os.remove(m.confusion_matrix_path_format.format(size="small"))
	os.remove(m.confusion_matrix_path_format.format(size="larger"))
