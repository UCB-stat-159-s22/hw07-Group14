
import warnings
import os
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import keras
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE



class Models:
	
	def __init__(self, **kwargs):
		self.data_split(kwargs["data"])
		self.autoencoder_path = kwargs["autoencoder_path"]
		self.encoder_path = kwargs["encoder_path"]
		self.classifier_path_format = kwargs["classifier_path_format"]
		self.projection_plot_path = kwargs["projection_plot_path"]
		self.cv_path = kwargs["cv_path"]
		self.roc_path_format = kwargs["roc_path_format"]
		self.confusion_matrix_path_format = kwargs["confusion_matrix_path_format"]
		self.test_performance_path = kwargs["test_performance_path"]
		self.max_epochs = kwargs["max_epochs"]
		self.verbose = kwargs["verbose"]
		self.encoder = None
	
	
	def data_split(self, data, test_size=0.2):
		"""
		Splits data into x with features and y with labels, and then again into 
		train and test. test_size controls the split proportions.
		
		Parameters
		-------
		data: pandas dataframe
			Columns representing gene (log) counts and rows representing cells.
			Last column contains the label, i.e. the organ the cell belongs to.
		
		Returns
		-------
		Nothing. Several objects are saved as arguments of the class instance.
		self.x: pandas dataframe
			Contains only features and all rows of data.
		self.x_train: pandas dataframe
			Contains only features and selection of rows inteded for training.
		self.x_test: pandas dataframe
			Contains only features and selection of rows inteded for testing.
		self.y: pandas dataframe
			Contains one hot encoded representation of labels for all rows.
		self.y_train: pandas dataframe
			Contains one hot encoded representation of labels, subset of rows
			intended for training.
		self.y_test: pandas dataframe
			Contains one hot encoded representation of labels, subset of rows
			intended for testing.
		self.y_raw: pandas dataframe
			Contains original label of data (not one hot encoded, i.e. only one
			column)
		self.organs: array-like
			Organs presented in the same order as in one hot encoded target dataframes
			(i.e. y, y_train, y_test).
		"""
		# Separate features from label
		x = data.iloc[:, :-1]
		y = data.iloc[:, -1]

		# One hot encoding of y labels
		# Order chosen by proportions in sample and kept in y_labels
		y_hot = pd.get_dummies(y)
		organs = y_hot.mean(axis=0).sort_values(ascending=False).index
		y_hot = y_hot[organs]
		y_hot.head()

		# 80:20 split shuffling data
		x_train, x_test, y_train, y_test = train_test_split(x, 
															y_hot, 
															test_size=test_size, 
															random_state=42, 
															shuffle=True, 
															stratify=y)
		
		# Save variables
		self.x = x 
		self.x_train = x_train 
		self.x_test = x_test 
		self.y = y_hot 
		self.y_train = y_train 
		self.y_test = y_test
		self.y_raw = y
		self.organs = organs
	
	
	def run_encoder(self):
		"""
		Initializes and trains autoencoder. It then extracts trained encoder
		from trained autoencoder and saves it.
		"""
		self.initialize_autoencoder()
		self.fit_autoencoder()
		self.autoencoder_performance()
		self.initialize_trained_encoder()
		self.save_encoder()
	
	
	def initialize_autoencoder(self, encoding_dim=32, dropout_value=0.2):
		"""
		Initializes autoencoder with the following characteristics:
		-Latent space of encoding_dim dimensions (default is 32).
		-1 hidden layer for encoder and one for decoder.
		-dropout before each fully connected layer.
		-L1 regularization used in fully connected layers.
		
		Encoder is defined later after training based on the first layers of the 
		autoencoder
		
		Parameters
		----------
		encoding_dim: int
			Number of dimensions in latent space (embedding).
		dropout_value: float
			Percentage of nodes output to be multiplied by zero in dropout layer.
			Nodes are chosen at random at each gradient step.
			
		Returns
		-------
		Nothing. It saves the following class instance argument:
		self.autoencoder: keras Model
			Untrained autoencoder
		It prints autoencoder architecture if class verbose argument is not 0.
		"""
		
		input_dim = self.x.shape[1]
		
		input_obj = keras.Input(shape=(input_dim,))
		dropout1 = keras.layers.Dropout(dropout_value)(input_obj)
		hidden1 = keras.layers.Dense(128, activation='relu', 
							   activity_regularizer=keras.regularizers.l1(10e-5))(dropout1)
		dropout2 = keras.layers.Dropout(dropout_value)(hidden1)
		encoded = keras.layers.Dense(encoding_dim, activation='relu', 
							   activity_regularizer=keras.regularizers.l1(10e-5))(dropout2)
		dropout3 = keras.layers.Dropout(dropout_value)(encoded)
		hidden2 = keras.layers.Dense(128, activation='relu')(dropout3)
		dropout4 = keras.layers.Dropout(dropout_value)(hidden2)
		decoded =keras.layers.Dense(input_dim, activation='sigmoid')(dropout4)
		
		self.autoencoder = keras.Model(input_obj, decoded)
		self.print_verbose("\nAutoencoder architecture:")
		if self.verbose:
			self.autoencoder.summary()
	
	
	def fit_autoencoder(self):
		"""
		Compiles and trains autoencoder using self.x_train for gradient step
		and self.x_test for validation.
		Several callbacks are used. End result is that best model in validation
		set is saved.
		
		Returns
		-------
		Nothing.
		self.autoencoder will be loaded with best model in validation set.
		Best autoencoder is also saved at self.autoencoder_path.
		"""
		
		# Configurations for improving training with the aim to reduce 
		# over-fitting and also allow to seek for a lower bias model.
		# Reference: https://stackoverflow.com/questions/48285129/saving-best-model-in-keras		
		early_stopping = keras.callbacks.EarlyStopping(
			monitor='val_loss', 
			patience=20, verbose=0,
			mode='min',
		)
		mcp_save = keras.callbacks.ModelCheckpoint(
			self.autoencoder_path, 
			save_best_only=True, 
			monitor='val_loss', 
			mode='min',
		)
		reduce_lr_loss = keras.callbacks.ReduceLROnPlateau(
			monitor='val_loss', 
			factor=0.5, 
			patience=10, 
			verbose=0, 
			min_delta=1e-4, 
			mode='min',
		)

		# Used MSE as loss
		self.autoencoder.compile(
			optimizer='adam', 
			loss='mean_squared_error',
		)

		self.autoencoder.fit(
			self.x_train, self.x_train,
			epochs=self.max_epochs,
			batch_size=256,
			shuffle=True,
			validation_data=(self.x_test, self.x_test),
			callbacks=[early_stopping, mcp_save, reduce_lr_loss],
			verbose=0,
		)
		
		# Load best
		self.autoencoder = keras.models.load_model(self.autoencoder_path, compile=False)
		self.autoencoder.compile(optimizer='adam', loss='mean_squared_error')
		
		self.print_verbose(f"Autoencoder saved at {self.autoencoder_path}\n")
	
	
	def autoencoder_performance(self):
		"""
			Evaluate autoencoder performance using loss function on test set.
		"""
		self.print_verbose("\nAutoencoder loss on test set")
		self.autoencoder.evaluate(
			self.x_test, 
			self.x_test, 
			verbose=self.verbose
		)
	
	
	def initialize_trained_encoder(self):
		"""
		Initializes encoder based on trained autoencoder.
		"""
		# Encoder
		encoder_output = self.autoencoder.layers[4].output
		encoder_input = self.autoencoder.input
		self.encoder = keras.Model(encoder_input, encoder_output)
		self.print_verbose("\nEncoder architecture:")
		if self.verbose:
			self.encoder.summary()
	
	
	def save_encoder(self):
		"""
		Saves encoder at self.encoder_path.
		"""
		# Save model
		self.encoder.compile(optimizer='adam', loss='mean_squared_error')
		keras.models.save_model(self.encoder, self.encoder_path, save_format='hdf5')
		self.print_verbose(f"Encoder saved at {self.encoder_path}")
	
	
	def load_encoder(self):
		"""
		Loads and compiles encoder from self.encoder_path.
		"""
		self.encoder = keras.models.load_model(self.encoder_path, compile=False)
		self.encoder.compile(optimizer='adam', loss='mean_squared_error')
	
	
	def plot_encoder_2d_and_compare(self):
		"""
		Plots two competing 2D representations of cells features.
		One is PCA (first two principal components).
		Second is t-SNE applied to encoder representation of data.
		
		Returns
		-------
		Nothing. Figure is saved at self.projection_plot_path.
		"""	
		
		# PCA
		pca32 = PCA(n_components=32)
		x_pca = pca32.fit_transform(self.x)
		
		# Encoder used to get input data's latent variables
		x_encoded = self.encoder.predict(self.x)
		x_encoded.shape
		
		# Encoder to 32 dim, then t-SNE to 2 dim
		tsne2 = TSNE(n_components=2, learning_rate=200, init='random', method='exact')
		x_encoded_tsne2 = tsne2.fit_transform(x_encoded)
		
		# Plot 2D representations
		fig, ax = plt.subplots(1, 2, figsize=(12, 6))
		
		x_dict = {
			'PCA': x_pca[:, :2],
			'Autoencoder to 32 dim,\nthen t-SNE to 2 dim': x_encoded_tsne2,
		}
		
		n = len(x_dict)

		for i, (title, x_plot) in enumerate(x_dict.items()):

			col = i
			ax_i = ax[col]

			sns.scatterplot(
				x=x_plot[:,0], y=x_plot[:,1],
				hue=self.y_raw,
				palette=sns.color_palette("hls", 2),
				legend="full",
				alpha=0.75,
				ax=ax_i)
			if i == n - 1:
				handles, labels = ax_i.get_legend_handles_labels()
			ax_i.get_legend().remove()
			ax_i.set_title(title)
			ax_i.set_xlabel('Component 1')
			if col == 0:
				ax_i.set_ylabel('Component 2')

		fig.legend(handles, labels, bbox_to_anchor=(0.9, 0.4), loc='lower center', ncol=1)
		fig.tight_layout()
		fig.subplots_adjust(right=0.8)

		plt.savefig(self.projection_plot_path, format='jpg')
		plt.show()
	
	
	def run_classifier(self):
		"""
		Initializes and trains classifier based on trained encoder for each 
		cross-validation fold.  Models and metrics are saved.
		"""
		self.initilize_classifiers()
		self.compute_and_save_cross_validation()
	
	
	def initilize_classifiers(self):
		"""
		Initializes small and larger classifiers.

		Returns
		-------
		Nothing. Models are saved in class instance arguments:
		self.nn_small: keras Model
		self.nn_larger: keras Model
		Architecture of both is optionally printed based on self.verbose
		"""
		# Small model quick validation
		self.nn_small = self.initilize_supervised_model('small')
		self.print_verbose("\nSmall classifier architecture:")
		if self.verbose:
			self.nn_small.summary()
		
		# Larger model quick validation
		self.nn_larger = self.initilize_supervised_model('larger')
		self.print_verbose("\nLarger classifier architecture:")
		if self.verbose:
			self.nn_larger.summary()
	
	
	def initilize_supervised_model(self, size):
		"""
		Initializes supervised model given by size. Only encoder part is trained.
		The architecture is controlled by the size parameter.
		
		Parameters
		----------
		size: string
			Either 'small' or 'larger'
		
		Returns
		-------
		nn_model: keras Model
			Supervised model built on top of encoder. Added layers are untrained.
		"""
		
		assert size in ['small', 'larger']
		
		if not self.encoder:
			self.load_encoder()
		
		# Add layers
		encoder_output = self.encoder.layers[-1].output
		encoder_input = self.encoder.input
		if size == 'small':
			# Small model
			dropout = keras.layers.Dropout(.2)(encoder_output)
			output = keras.layers.Dense(2, activation='softmax')(dropout)
			nn_model = keras.Model(encoder_input, output)
		elif size == 'larger':
			# Larger model
			dropout_output1 = keras.layers.Dropout(.2)(encoder_output)
			dense_output1 = keras.layers.Dense(16, activation='relu')(dropout_output1)
			dropout_output2 = keras.layers.Dropout(.2)(dense_output1)
			output = keras.layers.Dense(2, activation='softmax')(dropout_output2)
			nn_model = keras.Model(encoder_input, output)
		
		# Fix encoder parameters to avoid overfitting
		idx_not_train = [1, 2, 3, 4]
		for i in idx_not_train:
			nn_model.layers[i].trainable = False
		
		nn_model.compile(optimizer='adam', loss='categorical_crossentropy')

		return nn_model
	

	def compute_and_save_cross_validation(self):
		"""
		Computes and saves cross-valition results at self.cv_path.
		"""
		
		# Cross-validation results for small model: folds details
		res_cv_small = self.nn_cross_validation("small")
		res_cv_small
		
		# Cross-validation results for larger model: folds details
		res_cv_larger = self.nn_cross_validation("larger")
		res_cv_larger
		
		# Cross-validation aggregated results for both models
		res_cv = pd.concat((res_cv_small.iloc[:,1:].mean(axis=0), 
							res_cv_larger.iloc[:,1:].mean(axis=0)), 
						   axis=1).T.round(3)
		res_cv.index = ['small', 'larger']
		
		# Display
		if self.verbose:
			self.print_verbose("\nCross-validation results:")
			display(res_cv)
		
		# Save
		res_cv.to_csv(self.cv_path, index=True)
		self.print_verbose(f"\nCross-validation results saved at {self.cv_path}")
	
	
	def nn_cross_validation(self, size, n_split=4):
		"""
		Performs cross-validation on the classifier with architecture dictated by size.
		For each fold:
		- The classifier is initialized. Trained encoder parameters are not further trained.
		- The classifier last layers are trained using target data. Test set is not touched here.
		- Performance is measured and collected in a dataframe.
		
		Parameters
		----------
		size: string
			Either 'small' or 'larger'
		n_split: int
			Number of folds.
		
		Returns
		-------
		results: pandas dataframe
			One row per fold and one column per metric.
		
		Best fold models are constantly overwritten at self.classifier_path_format 
		(replacing appropriate size in string). At the end of the execution the 
		mentioned path will have the best model of the last fold.

		"""

		self.print_verbose(f"\nCross-validation for size = {size}")

		# Call backs
		# Early stopping
		early_stopping = keras.callbacks.EarlyStopping(
			monitor='val_loss', 
			patience=20, 
			verbose=0, 
			mode='min'
		)
		# Reduce learning rate on plateu
		reduce_lr_loss = keras.callbacks.ReduceLROnPlateau(
			monitor='val_loss', 
			factor=0.5, 
			patience=10, 
			min_delta=1e-4, 
			mode='min',
			verbose=0,
		)
		# Save best model by validation loss
		def mcp_save(size):
			return keras.callbacks.ModelCheckpoint(
			  self.classifier_path_format.format(size=size), 
			  save_best_only=True, 
			  monitor='val_loss', 
			  mode='min',
			)
		
		# Stratified k-fold is used
		kfold = StratifiedKFold(n_split, shuffle=True, random_state=0)
		
		# Reset index
		xx = self.x_train.copy().reset_index(drop=True)
		yy = self.y_train.copy().reset_index(drop=True)
		yy_ind = np.argmax(yy.values, axis=1)
		
		rows = []
		for i, (train_index, test_index) in enumerate(kfold.split(xx, yy_ind)):
			
			# Split
			xx_train, xx_test = xx.loc[train_index], xx.loc[test_index]
			yy_train, yy_test = yy.loc[train_index], yy.loc[test_index]
			
			# Train and save model
			nn_model = self.initilize_supervised_model(size)
			nn_model.fit(
				xx_train, yy_train,
				epochs=self.max_epochs,
				batch_size=256,
				shuffle=True,
				validation_data=(xx_test, yy_test),
				callbacks=[early_stopping, mcp_save(size), reduce_lr_loss],
				verbose=0,
			)
			
			# Load best model
			nn_model = keras.models.load_model(self.classifier_path_format.format(size=size), 
											   compile=False)
			nn_model.compile(optimizer='adam', loss='categorical_crossentropy')
			
			# Assess performance
			loss = nn_model.evaluate(xx_test, yy_test, verbose=0)
			y_true = yy_test.values
			y_score = nn_model.predict(xx_test)
			row_dict = self.compute_metrics(y_true, y_score)
			row_dict['categorical_crossentropy'] = loss
			rows.append(row_dict)
		
		# Collect performance results
		results = pd.DataFrame.from_dict(rows, orient='columns')
		results = results.reset_index().rename(columns={'index': 'fold'})
		
		return results
	
	
	def compute_metrics(self, y_true, y_score, average='macro'):
		"""
		Compute metrics.
		
		Returns
		-------
		Dictionary with metrics.
		"""
		assert y_true.shape == y_score.shape
		y_ind_pred = np.argmax(y_score, axis=1)
		y_ind_true = np.argmax(y_true, axis=1)
		accuracy = metrics.accuracy_score(y_ind_true, y_ind_pred)
		precision = metrics.precision_score(y_ind_true, y_ind_pred, average=average)
		recall = metrics.recall_score(y_ind_true, y_ind_pred, average=average)
		f1 = metrics.f1_score(y_ind_true, y_ind_pred, average=average)
		roc_auc = metrics.roc_auc_score(y_true, y_score, average=average)
		return {
		  "accuracy": accuracy,
		  "precision": precision,
		  "recall": recall,
		  "f1": f1,
		  "roc_auc": roc_auc,
		}
	
	
	def run_assessment(self):
		"""
		Measure performance of classifiers (both small and larger).
		Saves test set performance, confusion matrices heatmap and roc curves plot.
		"""
		self.load_classifiers()
		self.test_performance()
		self.plot_confusion_matrices()
		self.plot_roc_curves()
	
	
	def load_classifiers(self):
		"""
		Loads and compiles small and larger model to self.nn_small and self.nn_larger.
		"""
		# Load last model trained in cross-validation
		# Training needs a validation set so any fold model is OK
		self.nn_small = keras.models.load_model(self.classifier_path_format.format(size='small'), 
												compile=False)
		self.nn_larger = keras.models.load_model(self.classifier_path_format.format(size='larger'), 
												 compile=False)
		self.nn_small.compile(optimizer='adam', loss='categorical_crossentropy')
		self.nn_larger.compile(optimizer='adam', loss='categorical_crossentropy')
	
	
	def get_truth_and_score(self):
		"""
		Computes inputs needed for several performance tables and plots.
		
		Returns
		-------
		y_true: numpy array
			y_test values (one hot encoded, i.e. two columns)
		y_true_indices: numpy array
			one column version of y_true containing indices instead
		y_score_small: numpy array
			small model prediction of self.x_test
		y_score_larger: numpy array
			larger model prediction of self.x_test
		"""
		# Truth and predictions
		y_true = self.y_test.values
		y_true_indices = np.argmax(y_true, axis=1)
		y_score_small = self.nn_small.predict(self.x_test)
		y_score_larger = self.nn_larger.predict(self.x_test)
		return y_true, y_true_indices, y_score_small, y_score_larger
	
	
	def test_performance(self):
		"""
		Computs performance in test set for both models.
		Saves results at self.test_performance_path
		"""

		# Get inputs
		y_true, y_true_indices, y_score_small, y_score_larger = self.get_truth_and_score()
		
		# Performance on test set
		metrics_small = self.compute_metrics(y_true, y_score_small)
		metrics_larger = self.compute_metrics(y_true, y_score_larger)
		res = pd.DataFrame.from_dict([metrics_small, metrics_larger], 
									 orient='columns').round(3)
		res.index = ['small', 'larger']
		
		# Display
		if self.verbose:
			self.print_verbose("\nTest set performance:")
			display(res)
		
		# Save
		res.to_csv(self.test_performance_path, index=True)
	
	
	def plot_confusion_matrices(self):
		"""
		Computes confusion matrices for both models.
		Plots a heatmap of each matrix and saves it at self.confusion_matrix_path_format
		(replacing size appropriately).
		"""
		
		# Get inputs
		y_true, y_true_indices, y_score_small, y_score_larger = self.get_truth_and_score()
		
		# Confusion matrix
		conf_matrix_small = metrics.confusion_matrix(y_true_indices, 
													 np.argmax(y_score_small, axis=1))
		conf_matrix_larger = metrics.confusion_matrix(y_true_indices, 
													  np.argmax(y_score_larger, axis=1))
		
		# Plot matrix and save
		self.plot_confusion_matrix_model("small", conf_matrix_small)
		self.plot_confusion_matrix_model("larger", conf_matrix_larger)
	
	
	def plot_confusion_matrix_model(self, size, conf_matrix, title=True):
		"""
		Generates and saves heatmap of confusion matrix for specified model.
		
		Parameters
		----------
		size: string
			Either 'small' or 'larger'
		conf_matrix: numpy array
			Confusion matrix. Prediction at columns and truth at rows.
		tittle: boolean
			True for including a title to the figure.
			
		Returns
		-------
		Nothing. Saves the heatmap at self.confusion_matrix_path_format according
		to size.
		"""
		
		self.print_verbose(f"\nConfusion Matrix {size.title()}")
		
		plt.figure(figsize=(8, 6))
		sns.heatmap(
			conf_matrix, 
			cmap='Blues', 
			annot=True, 
			fmt='g',
			yticklabels=self.organs, 
			xticklabels=self.organs,
		)
		if title:
			plt.title(f"Confusion Matrix {size.title()}")
		plt.xlabel("Predicted", fontsize=13)
		plt.ylabel("Actual", fontsize=13)
		extension = self.confusion_matrix_path_format.split('.')[-1]
		plt.savefig(self.confusion_matrix_path_format.format(size=size), 
					format=extension)
		plt.show()
	
	
	def plot_roc_curves(self):
		"""
		Plots and saves roc curves for both models.
		"""
		
		# Get inputs
		y_true, y_true_indices, y_score_small, y_score_larger = self.get_truth_and_score()
		
		# Plot
		self.plot_roc_curve_model("small", y_true_indices, y_score_small)
		self.plot_roc_curve_model("larger", y_true_indices, y_score_larger)
	
	
	def plot_roc_curve_model(self, size, y_true_indices, y_score, title=True):
		"""
		Plots and saves roc curve for model specified by size.
		
		Parameters
		----------
		size: string
			Either 'small' or 'larger'
		y_true_indices: numpy array
			One column version of y_true containing indices instead
		y_score: numpy array
			Model prediction
			
		Returns
		-------
		Nothing. Saves the plot at self.roc_path_format
		"""

		self.print_verbose(f"\nROC curve {size.title()}")
				
		plt.figure(figsize=(7, 6))
		
		# Compute and plot fpr and tpr per selected labels/indices
		for ind, organ in enumerate(self.organs):
			y_score_ind = y_score[:, ind]
			y_true_ind = y_true_indices == ind
			fpr, tpr, threshold = metrics.roc_curve(y_true_ind, y_score_ind)
			roc_auc = metrics.auc(fpr, tpr)
			plt.plot(fpr, tpr, label = f"{organ} (AUC={roc_auc:.2f})")
		
		if title:
			plt.title(f'Receiver Operating Characteristic {size.title()}')
		plt.legend(loc='lower right')
		plt.plot([0, 1], [0, 1], 'r--')
		plt.xlim([0, 1])
		plt.ylim([0, 1])
		plt.ylabel('True Positive Rate')
		plt.xlabel('False Positive Rate')
		extension = self.roc_path_format.split('.')[-1]
		plt.savefig(self.roc_path_format.format(size=size), format=extension)
		plt.show()

	def print_verbose(self, s):
		"""
		Print string s if self.verbose is not 0 or False.
		
		Parameters
		----------
		s: string
			Any string to possibly print.
		
		Returns
		-------
		Nothing. Might print according to self.verbose.
		"""
		if self.verbose:
			print(s)