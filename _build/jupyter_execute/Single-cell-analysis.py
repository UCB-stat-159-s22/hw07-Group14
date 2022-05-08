#!/usr/bin/env python
# coding: utf-8

# # Final Project STAT 259: Single Cell Sequencing Analysis

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import keras
from keras import layers
from keras import regularizers
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import save_model, load_model


# ## Read data

# In[2]:


def setup_data(name):
    """
    ...
    """
    file = f"data/{name}-counts.csv"
    counts = pd.read_csv(file)
    counts.rename(columns={'Unnamed: 0':'index'}, inplace=True)
    counts.set_index("index", inplace=True)
    counts = counts.T
    counts.insert(0, 'cell_label', name)
    return counts


# In[3]:


organs = [
    "Kidney",
    "Liver",
]


# In[4]:


assert len(organs) == 2


# ## Feature filter

# In[5]:


data = pd.DataFrame()
for organ in organs:
    data = data.append(setup_data(organ))
data.shape


# In[6]:


# Remove gene counts with zero variance
std = data.std(axis=0, numeric_only=True)
final_features = list(std[std != 0].index)
data = data[final_features + ["cell_label"]]
data.shape


# In[7]:


std = std[final_features]


# In[8]:


mu = data.groupby(data["cell_label"]).mean().T


# In[9]:


score = np.abs((mu[organs[0]] - mu[organs[1]])/std).sort_values(ascending=False)
score.head(20)


# In[107]:





# In[10]:


# Choose subset of gene counts
num_genes = 1000
selected_features = list(score.iloc[:num_genes].index)
data = data[selected_features + ["cell_label"]]
data.shape


# ## Feature preprocessing

# In[11]:


data.iloc[:, :-1] = np.log(data.iloc[:,:-1] + 1)


# In[12]:


filepath = "processed_data/data.csv"
data.to_csv(filepath, index=True)


# # Lower-dimension representation of the cells

# In[13]:


data = pd.read_csv(filepath, index_col=0)


# In[14]:


data


# In[15]:


x = data.iloc[:, :-1]
y = data.iloc[:, -1]


# In[16]:


# One hot encoding of y labels
# Order chosen by proportions in sample and kept in y_labels
y_hot = pd.get_dummies(y)
y_labels = y_hot.mean(axis=0).sort_values(ascending=False).index
y_hot = y_hot[y_labels]
y_hot.head()


# In[17]:


# 80:20 split shuffling data
x_train, x_test, y_train, y_test = train_test_split(x, y_hot, test_size=0.2, random_state=42, shuffle=True, stratify=y)


# In[18]:


# Shape validation for x
x.shape, x_train.shape, x_test.shape


# In[19]:


# Shape validation for y
y.shape, y_train.shape, y_test.shape


# ## Autoencoder

# In[20]:


# Latent space of 32 dimensions
# 1 hidden layer for encoder and one for decoder
# L1 regularization used in fully connected layers
# Encoder will be defined later after training

encoding_dim = 32
input_dim = x.shape[1]

input_obj = keras.Input(shape=(input_dim,))
dropout1 = layers.Dropout(.2)(input_obj)
hidden1 = layers.Dense(128, activation='relu', 
                       activity_regularizer=regularizers.l1(10e-5))(dropout1)
dropout2 = layers.Dropout(.2)(hidden1)
encoded = layers.Dense(encoding_dim, activation='relu', 
                       activity_regularizer=regularizers.l1(10e-5))(dropout2)
dropout3 = layers.Dropout(.2)(encoded)
hidden2 = layers.Dense(128, activation='relu')(dropout3)
dropout4 = layers.Dropout(.2)(hidden2)
decoded = layers.Dense(input_dim, activation='sigmoid')(dropout4)

autoencoder = keras.Model(input_obj, decoded)
autoencoder.summary()


# In[21]:


# Configurations for improving training with the aim to reduce over-fitting 
# and also allow to seek for a lower bias model.
# Reference: https://stackoverflow.com/questions/48285129/saving-best-model-in-keras
earlyStopping = EarlyStopping(monitor='val_loss', patience=20, verbose=0, mode='min')
mcp_save = ModelCheckpoint('models/autoencoder_best.hdf5', save_best_only=True, monitor='val_loss', mode='min')
reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0, min_delta=1e-4, mode='min')


# In[22]:


# Used MSE as loss
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

autoencoder.fit(
    x_train, x_train,
    epochs=500,
    batch_size=256,
    shuffle=True,
    validation_data=(x_test, x_test),
    callbacks=[earlyStopping, mcp_save, reduce_lr_loss],
    verbose=0,
)


# In[23]:


# Load best
autoencoder = load_model('models/autoencoder_best.hdf5', compile=False)
autoencoder.compile(optimizer='adam', loss='mean_squared_error')


# In[24]:


# Evaluate autoencoder performance
autoencoder.evaluate(x_test, x_test)


# In[25]:


# Encoder
encoder_output = autoencoder.layers[4].output
encoder_input = autoencoder.input
encoder = keras.Model(encoder_input, encoder_output)
encoder.summary()


# In[26]:


# Save model
encoder.compile(optimizer='adam', loss='mean_squared_error')
save_model(encoder, 'models/encoder_best.hdf5', save_format='hdf5')


# In[27]:


# PCA
pca32 = PCA(n_components=32)
x_pca = pca32.fit_transform(x)


# In[28]:


# t-SNE
tsne2 = TSNE(n_components=2, learning_rate=200, init='random', method='exact')
x_tsne2 = tsne2.fit_transform(x_pca)


# In[127]:


# Encoder used to get input data's latent variables
x_encoded = encoder.predict(x)
x_encoded.shape


# In[128]:


# Encoder to 32 dim, then t-SNE to 2 dim
x_encoded_tsne2 = tsne2.fit_transform(x_encoded)


# In[134]:


# Plot 2D representations
ncols = 2
fig, ax = plt.subplots(1, ncols, figsize=(12, 6))

x_dict = {
    'PCA': x_pca[:, :2],
    #'t-SNE': x_tsne2,
    'Autoencoder to 32 dim,\nthen t-SNE to 2 dim': x_encoded_tsne2,
}

n = len(x_dict)

for i, (title, x_plot) in enumerate(x_dict.items()):

    col = i
    ax_i = ax[col]

    sns.scatterplot(
        x=x_plot[:,0], y=x_plot[:,1],
        hue=y,
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

plt.savefig('figures/projection.jpg', format='jpg')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




