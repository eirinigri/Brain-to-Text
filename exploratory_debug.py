#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis & Visualization
# This notebook consolidates the analysis scripts for the Brain-to-Text project, including data visualization, correlation analysis, and advanced unsupervised learning.

# In[1]:


# --- Setup Environment ---
get_ipython().system('pip install h5py pandas plotly scikit-learn hdbscan ipykernel nbformat>=4.2.0 ipywidgets')


# In[2]:


import h5py
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
try:
    from sklearn.cluster import HDBSCAN
except ImportError:
    try:
        import hdbscan as HDBSCAN # specialized lib
    except ImportError:
         print("HDBSCAN not installed, will skip or use dummy.")
         HDBSCAN = None

# Configuration
DATA_DIR = r"c:\Users\pswmi\myproj\Brain-to-Text\t15_copyTask_neuralData\hdf5_data_final"
SUBFOLDER = "t15.2023.08.11"
FILE_PATH = os.path.join(DATA_DIR, SUBFOLDER, "data_train.hdf5")


# In[7]:


# Peek at the HDF5 file structure
import h5py
with h5py.File(FILE_PATH, "r") as f:
    sample_key = list(f.keys())[0]
    print("Keys in trial:", list(f[sample_key].keys()))
    print("Sample seq_class_ids:", f[sample_key]['seq_class_ids'][:10])


# ## 1. General Data Visualization

# In[3]:


def load_data(file_path, num_samples=100):
    neural_data = []
    phoneme_lens = []
    input_lens = []

    with h5py.File(file_path, "r") as f:
        keys = sorted(list(f.keys()))
        subset_keys = keys[:num_samples]
        for k in subset_keys:
            trial = f[k]
            if 'input_features' in trial and 'seq_class_ids' in trial:
                feat = trial['input_features'][:]
                neural_data.append(feat)
                input_lens.append(feat.shape[0])
                phoneme_lens.append(trial['seq_class_ids'].shape[0])
    return neural_data, input_lens, phoneme_lens

print(f"Loading data from {FILE_PATH}...")
try:
    neural_data, input_lens, phoneme_lens = load_data(FILE_PATH, num_samples=200)

    # Subplots
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"type": "xy"}, {"type": "xy"}],
               [{"type": "xy", "colspan": 2}, None]],
        subplot_titles=("Input Sequence Length Distribution", "Target Phoneme Length Distribution", "PCA of Averaged Neural Activity")
    )
    fig.add_trace(go.Histogram(x=input_lens, name="Input Frames", nbinsx=20, marker_color='#636EFA'), row=1, col=1)
    fig.add_trace(go.Histogram(x=phoneme_lens, name="Phonemes", nbinsx=20, marker_color='#EF553B'), row=1, col=2)

    # PCA
    mean_vectors = np.array([np.mean(x, axis=0) for x in neural_data])
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(mean_vectors)

    fig.add_trace(go.Scatter(x=pca_result[:,0], y=pca_result[:,1], mode='markers', name='Trials'), row=2, col=1)
    fig.update_layout(height=800, title_text="Data Overview")
    fig.show()

    # Heatmap
    sample_data = neural_data[0].T
    hm = go.Figure(data=go.Heatmap(z=sample_data, colorscale='Magma'))
    hm.update_layout(title="Neural Activity Heatmap (Trial 0)", height=500)
    hm.show()

except Exception as e:
    print(f"Error: {e}")


# ## 2. Cluster-to-Phoneme Correlation Analysis

# In[9]:


VOCAB = ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH', 'EH', 'ER', 'EY', 'F', 'G', 'HH', 'IH', 'IY', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW', 'OY', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z', 'ZH', '|']
IDX_TO_PHONEME = {i+1: v for i, v in enumerate(VOCAB)}
IDX_TO_PHONEME[0] = "BLANK"

def load_data_and_align(file_path, num_trials=50):
    all_frames = []
    all_labels = []
    with h5py.File(file_path, "r") as f:
        keys = sorted(list(f.keys()))[:num_trials]
        for k in keys:
            trial = f[k]
            if 'input_features' in trial and 'seq_class_ids' in trial:
                feat = trial['input_features'][:]
                phonemes = trial['seq_class_ids'][:]
                T = feat.shape[0]
                L = phonemes.shape[0]
                if L == 0: continue
                # Uniform Alignment
                frames_per_phoneme = T / L
                for t in range(T):
                    idx = min(int(t // frames_per_phoneme), L-1)
                    all_frames.append(feat[t])
                    all_labels.append(phonemes[idx])
    return np.array(all_frames), np.array(all_labels)

print("Running Correlation Analysis...")
X, Y = load_data_and_align(FILE_PATH, num_trials=50)

# Filter out BLANK tokens (padding)
mask = Y != 0
X_filtered = X[mask]
Y_filtered = Y[mask]

print(f"Frames after filtering BLANK: {len(Y_filtered)}")
print(f"Unique phoneme IDs: {np.unique(Y_filtered)}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

df = pd.DataFrame({'Cluster': clusters, 'TokenID': Y})
df['Phoneme'] = df['TokenID'].map(IDX_TO_PHONEME)
heatmap_data = pd.crosstab(df['Cluster'], df['Phoneme'], normalize='index')

fig = go.Figure(data=go.Heatmap(z=heatmap_data.values, x=heatmap_data.columns, y=[f"State {i}" for i in heatmap_data.index], colorscale='Viridis'))
fig.update_layout(title="Neural State to Phoneme Correlation", height=600)
fig.show()

print(f"After filter - Unique phoneme IDs: {np.unique(Y)}")


# ## 3. Advanced Unsupervised Analysis (Manifold Learning)

# In[5]:


MAX_FRAMES_FOR_TSNE = 2000 
print("Running Advanced Analysis...")
# Reuse X from previous cell if available, else load
if 'X' not in locals():
    # Simple loader if not already loaded
    def load_frames_simple(file_path, n=5):
        frames = []
        with h5py.File(file_path, "r") as f:
            for k in sorted(list(f.keys()))[:n]:
                if 'input_features' in f[k]: frames.append(f[k]['input_features'][:])
        return np.vstack(frames)
    X = load_frames_simple(FILE_PATH)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Subsample for t-SNE
if X_scaled.shape[0] > MAX_FRAMES_FOR_TSNE:
    indices = np.random.choice(X_scaled.shape[0], MAX_FRAMES_FOR_TSNE, replace=False)
    X_tsne_input = X_scaled[indices]
else:
    X_tsne_input = X_scaled

# t-SNE
tsne = TSNE(n_components=2, perplexity=30, init='pca', learning_rate='auto')
X_embedded = tsne.fit_transform(X_tsne_input)

fig = px.scatter(x=X_embedded[:,0], y=X_embedded[:,1], title="t-SNE of Neural Manifold", opacity=0.6)
fig.show()

