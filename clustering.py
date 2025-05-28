import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE



# Load CMAPSS FD001 Dataset
def load_cmapss(file_name):
    df = pd.read_csv(file_name,sep='\s+',header=None)
    df.dropna(axis=1, inplace=True)  # Remove empty columns
    df.columns = ['unit', 'time','setting_1', 'setting_2', 'setting_3'] + [f's{i}' for i in range(1, 22)]
    #df['RUL'] = df.groupby('unit')['time'].transform(max) - df['time']
    df.drop(['setting_1','setting_2','setting_3','s1','s5','s6','s10','s16','s18','s19'],axis=1,inplace=True)
    return df

df = load_cmapss('train_FD001.txt')

from sklearn.preprocessing import MinMaxScaler

# Initialize the scaler
scaler = MinMaxScaler()

# Create an empty DataFrame for the normalized data
df_norm = pd.DataFrame()

# Iterate over each engine ID and normalize the sensor values
for engine_id in df['unit'].unique():
    # Filter the data for the current engine
    engine_data = df[df['unit'] == engine_id]
    
    # Normalize the sensor data (columns from index 2 to -1, assuming these are sensor columns)
    normalized_sensors = scaler.fit_transform(engine_data.iloc[:, 2:-1])
    
    # Create a DataFrame with normalized sensor data
    normalized_df = pd.DataFrame(normalized_sensors, columns=df.columns[2:-1])
    
    # Add the 'engine_id', 'RUL', and 'time_cycles' columns back
    normalized_df['unit'] = engine_data['unit'].values
    normalized_df['time'] = engine_data['time'].values
    
    # Append the normalized data for this engine to the final DataFrame
    df_norm = pd.concat([df_norm, normalized_df], ignore_index=True)
    
for engine_id in [1,10,20,30,40,50,60,70,80,90,100]:  # Visualize only some engines for clarity
    sensor_data = df_norm[df_norm.unit==engine_id].iloc[:,2:]  # Remove unit & cycle columns


    # Apply DBSCAN
    eps = 1.17# Adjust based on trial and error
    min_samples = 3
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(sensor_data)

    # Append cluster labels to the dataframe
    sensor_data["cluster"] = labels


    # Apply t-SNE for dimensionality reduction
    tsne = TSNE(n_components=2,random_state=42, n_iter=1000)
    data_tsne = tsne.fit_transform(sensor_data)

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c=sensor_data["cluster"], cmap="viridis", alpha=0.6)
    plt.colorbar(scatter, label="Cluster Label")
    plt.title(f"Engine {engine_id}: DBSCAN Clustering on CMAPSS Data (t-SNE Projection)")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.show()