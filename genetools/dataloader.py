
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np


class DataPreprocessing:
	"""
	Reads, preprocesses, and saves data needed for supervised learning task.
	"""
	
	def __init__(self, **kwargs):
		
		# Assertions
		assert len(kwargs["organs"]) == 2
		
		# Attributes
		self.organs = kwargs["organs"]
		self.y_label = kwargs["y_label"]
		self.input_path_format = kwargs["input_path_format"]
		self.output_path = kwargs["output_path"]
		self.bool_filter_zero_variance = kwargs["bool_filter_zero_variance"]
		self.bool_filter_mean_diff = kwargs["bool_filter_mean_diff"]
		self.bool_take_log = kwargs["bool_take_log"]
		self.bool_save = kwargs["bool_save"]
		self.num_genes = kwargs["num_genes"]
	
	
	def run(self):
		"""
		The following steps can be executed:
		- Reads the data for specified organs.
		- Filters features with zero variance.
		- Keeps NUM_GENES features (gene counts) based on 
		  mean difference between organs.
		- Takes log.
		- Saves the model.
		The arguments of the class can be used to control which steps are ran
		and which are not.

		Returns
		-------
		data: pandas dataframe
			Columns representing gene counts and rows representing cells.
			Last column contains the label, i.e. the organ the cell belongs to.
		"""	
		
		data = self.load_input_data()
		if self.bool_filter_zero_variance:
			data = self.filter_features_with_zero_variance(data)
		if self.bool_filter_mean_diff:
			data = self.filter_features_mean_diff_score(data)
		if self.bool_take_log:
			data = self.take_log(data)
		if self.bool_save:
			self.save_data(data)
		return data
	
	
	def load_input_data(self):
		"""
		Reads and appends all dataframes located at self.input_path_format
		after replacing in the path each organ present in self.organs
		
		Returns
		-------
		data: pandas dataframe
			Columns representing gene counts and rows representing cells.
			Last column contains the label, i.e. the organ the cell belongs to.
		"""
		print("Loading data")
		data = pd.DataFrame()
		for organ in self.organs:
			data = data.append(self.load_data_organ(organ))
		return data
	
	
	def load_data_organ(self, organ):
		"""
		Reads csv file at the path obtained by replacing the organ given
		in self.input_path_format.
		
		Parameters
		---------
		organ: string 
			Example: 'liver' or 'kidney'
		
		Returns
		-------
		data: pandas dataframe
			Columns representing gene counts and rows representing cells.
			Last column contains the label, i.e. the organ the cell belongs to.
		"""
		data = pd.read_csv(self.input_path_format.format(organ=organ))
		data.rename(columns={'Unnamed: 0': 'index'}, inplace=True)
		data.set_index("index", inplace=True)
		data = data.T
		data.insert(0, self.y_label, organ)
		return data
	
	
	def filter_features_with_zero_variance(self, data):
		"""
		Filters feature columns with zero variance.
		
		Parameters
		----------
		data: pandas dataframe
		
		Returns
		-------
		data: pandas dataframe
		"""
		print("Filtering features with zero variance")
		std = data.std(axis=0, numeric_only=True)
		final_features = list(std[std != 0].index)
		data = data[final_features + [self.y_label]]
		return data
	
	
	def filter_features_mean_diff_score(self, data):
		"""
		Filter features based on difference of group means
		divided total standard deviation. 
		Keeps the top self.num_genes features with the highest score.
		
		Parameters
		----------
		data: pandas dataframe
		
		Returns
		-------
		data: pandas dataframe
		"""
		print("Filtering features based on difference of group means")
		
		# Compute mean difference score
		std = data.std(axis=0, numeric_only=True)
		mu = data.groupby(self.y_label).mean().T
		stat = np.abs((mu[self.organs[0]] - mu[self.organs[1]])/std)
		score = stat.sort_values(ascending=False)

		# Choose subset of gene counts
		selected_features = list(score.iloc[:self.num_genes].index)
		data = data[selected_features + [self.y_label]]
		return data
	
	
	def take_log(self, data):
		"""
		Compute logarithm of features (gene counts).

		Parameters
		----------
		data: pandas dataframe
		
		Returns
		-------
		data: pandas dataframe
		"""
		data.iloc[:, :-1] = np.log(data.iloc[:, :-1] + 1)
		return data
	
	
	def save_data(self, data):
		"""
		Save the data in self.output_path.
		
		Parameters
		----------
		data: pandas dataframe
		
		Returns
		-------
		Nothing. The dataframe is saved.
		"""
		data.to_csv(self.output_path, index=True)
	
	
	def load_processed_data(self):
		"""
		Load the data located in self.output_path.
		
		Returns
		-------
		data: pandas dataframe
		"""
		data = pd.read_csv(self.output_path, index_col=0)
		return data
