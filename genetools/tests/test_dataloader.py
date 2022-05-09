
import pytest
import pandas as pd
import conftest as cf


def test_data(data):
	"""
	Validate that processed data has:
	- At least one row.
	- Feasible number of columns.
	- Correct number of organs in y label.
	"""
	
	assert isinstance(data, pd.DataFrame)
	assert data.shape[0] > 0
	assert data.shape[1] == cf.NUM_GENES + 1
	assert data.iloc[:, -1].nunique() == 2



