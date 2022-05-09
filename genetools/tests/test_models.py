
import os
import pytest
import pandas as pd
import conftest as cf


def test_check_xs_shape(m):
	"""
	Validate that x, x_train, and x_test have feasible dimensions.
	"""
	
	assert isinstance(m.x, pd.DataFrame)
	assert isinstance(m.x_train, pd.DataFrame)
	assert isinstance(m.x_test, pd.DataFrame)
	assert m.x.shape[1] == cf.NUM_GENES
	assert m.x_train.shape[1] == cf.NUM_GENES
	assert m.x_test.shape[1] == cf.NUM_GENES
	assert m.x.shape[0] == m.x_train.shape[0] + m.x_test.shape[0]

	
def test_check_ys_shape(m):
	"""
	Validate that y, y_train, and y_test have feasible dimensions.
	"""
	
	assert isinstance(m.y, pd.DataFrame)
	assert isinstance(m.y_train, pd.DataFrame)
	assert isinstance(m.y_test, pd.DataFrame)
	assert m.y.shape[1] == 2
	assert m.y_train.shape[1] == 2
	assert m.y_test.shape[1] == 2
	assert m.y.shape[0] == m.y_train.shape[0] + m.y_test.shape[0]
	assert m.y.shape[0] == m.x.shape[0]


def test_encoder_exists(run_encoder, m):
	"""
	Validate autoencoder and encoder were saved.
	"""
	
	assert os.path.isfile(m.autoencoder_path)
	assert os.path.isfile(m.encoder_path)


def test_2d_plot(run_2d_plot, m):
	"""
	Validate that 2D plot with projections was saved.
	"""

	assert os.path.isfile(m.projection_plot_path)


def test_classifiers_exist(run_classifier, m):
	"""
	Validate that all classifiers were saved in appropriate files.
	"""
	
	assert os.path.isfile(m.classifier_path_format.format(size="small"))
	assert os.path.isfile(m.classifier_path_format.format(size="larger"))


def test_classifiers_performance(run_assessment, m):
	"""
	Validate that all fitted models have at least 65% accuracy.
	"""
	
	results_test = pd.read_csv(m.test_performance_path)
	assert (results_test.accuracy >= 0.65).all()
