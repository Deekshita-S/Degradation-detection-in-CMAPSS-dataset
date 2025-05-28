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


healthy_data = df_norm.groupby('unit').head(60).drop(['unit','time'],axis=1) # Based on literature, degradation starts after time cycle 70


from sklearn.svm import OneClassSVM

ocsvm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.05)  # nu can be tuned
ocsvm.fit(healthy_data)

df_norm_pred=df_norm.drop(['unit','time'],axis=1)

pred_labels = ocsvm.predict(df_norm_pred)  # +1 for inliers (normal), -1 for outliers (anomalies)
df_norm['ocsvm_pred'] = pred_labels


degradation_onset={} # key: unit / engine_id, value=degradation onset time
for i in df_norm.unit.unique():
    degradation_onset[i]=df_norm[(df_norm['ocsvm_pred']==-1) & (df.unit==i)].iloc[7,-2]


print(degradation_onset)

