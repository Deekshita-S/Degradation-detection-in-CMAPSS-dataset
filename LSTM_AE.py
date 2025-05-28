import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn


# Load CMAPSS FD001 Dataset
def load_cmapss(file_name):
    df = pd.read_csv(file_name,sep='\s+',header=None)
    df.dropna(axis=1, inplace=True)  # Remove empty columns
    df.columns = ['unit', 'time','setting_1', 'setting_2', 'setting_3'] + [f's{i}' for i in range(1, 22)]
    #df['RUL'] = df.groupby('unit')['time'].transform(max) - df['time']
    df.drop(['setting_1','setting_2','setting_3','s1','s5','s6','s10','s16','s18','s19'],axis=1,inplace=True)
    return df

data = load_cmapss('train_FD001.txt')

data.head()


## Moving avg per unit to smooth the data

def normalize_per_engine(df):
    df_norm = df.copy()
    for engine_id in df['unit'].unique():
        engine_data = df[df['unit'] == engine_id]
        scaler = StandardScaler()
        df_norm.loc[df['unit'] == engine_id, df.columns[2:]] = scaler.fit_transform(engine_data.iloc[:, 2:])
    return df_norm

data=normalize_per_engine(data)

## splitting data into training and validation.

healthy=data.groupby('unit').head(60)
train_df,val_df=train_test_split(healthy,test_size=0.2,random_state=42)


# Genreate seq to be fed as input to LSTMs
def create_sequences_per_engine(df, seq_length=5):
    sequences = []
    for engine_id in df['unit'].unique():
        engine_data = df[df['unit'] == engine_id].iloc[:, 2:].values  # Exclude unit and cycle columns
        for i in range(len(engine_data) - seq_length):
            sequences.append(engine_data[i:i + seq_length])
    return torch.tensor(sequences, dtype=torch.float32)


train_df = create_sequences_per_engine(train_df)
val_df = create_sequences_per_engine(val_df)



# Model

class Encoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim):
        super(Encoder, self).__init__()
        self.seq_len, self.n_features = seq_len, n_features
        self.hidden1_dim, self.hidden2_dim = 32 * embedding_dim, 8 * embedding_dim

        self.lstm1 = nn.LSTM(
            input_size=n_features,
            hidden_size=self.hidden1_dim,
            num_layers=1,
            batch_first=True
        )
        
        self.layer_norm1 = nn.LayerNorm(self.hidden1_dim)  # Layer Normalization after lstm1

        self.lstm2 = nn.LSTM(
            input_size=self.hidden1_dim,
            hidden_size=self.hidden2_dim,
            num_layers=1,
            batch_first=True
        )
        
        self.layer_norm2 = nn.LayerNorm(self.hidden2_dim)
        
        self.lstm3 = nn.LSTM(
            input_size=self.hidden2_dim,
            hidden_size=embedding_dim,
            num_layers=1,
            batch_first=True
        )
        
        self.layer_norm3 = nn.LayerNorm(embedding_dim)  # Layer Normalization after lstm2

        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        x = x.reshape((1, self.seq_len, self.n_features))
        x, (_, _) = self.lstm1(x)
        x = self.layer_norm1(x)  # Apply layer normalization
        x = self.dropout(x)
        x, (_, _) = self.lstm2(x)
        x = self.layer_norm2(x)  # Apply layer normalization
        x = self.dropout(x)
        x, (hidden_n, _) = self.lstm3(x)
        x = self.layer_norm3(x)  # Apply layer normalization
        x = self.dropout(x)
       
        return hidden_n  # Return the last hidden state

class Decoder(nn.Module):
    def __init__(self, seq_len, input_dim, n_features):
        super(Decoder, self).__init__()
        self.seq_len, self.input_dim = seq_len, input_dim
        self.hidden1_dim, self.hidden2_dim, self.n_features = 8 * input_dim, 32 *input_dim, n_features

        self.rnn1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=input_dim,
            num_layers=1,
            batch_first=True
        )
        
        self.layer_norm1 = nn.LayerNorm(input_dim)  # Layer Normalization after rnn1

        self.rnn2 = nn.LSTM(
            input_size=input_dim,
            hidden_size=self.hidden1_dim,
            num_layers=1,
            batch_first=True
        )

        self.layer_norm2 = nn.LayerNorm(self.hidden1_dim)  # Layer Normalization after rnn2

        self.rnn3 = nn.LSTM(
            input_size=self.hidden1_dim,
            hidden_size=self.hidden2_dim,
            num_layers=1,
            batch_first=True
        )

        self.layer_norm3 = nn.LayerNorm(self.hidden2_dim)
        
        self.dropout = nn.Dropout(0.4)
        
        self.output_layer = nn.Linear(self.hidden2_dim, self.n_features)

    def forward(self, x):
        x = x.repeat(1, self.seq_len, 1)
        x, _ = self.rnn1(x)
        x = self.layer_norm1(x)  # Apply layer normalization
        x = self.dropout(x)
        x, _ = self.rnn2(x)
        x = self.layer_norm2(x)  # Apply layer normalization
        x = self.dropout(x)
        x, _ = self.rnn3(x)
        x = self.layer_norm3(x)  # Apply layer normalization
        x = self.dropout(x)
        return self.output_layer(x)

class RecurrentAutoencoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim):
        super(RecurrentAutoencoder, self).__init__()
        self.encoder = Encoder(seq_len, n_features, embedding_dim)
        self.decoder = Decoder(seq_len, embedding_dim, n_features)
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    



## Training
model = RecurrentAutoencoder(5, 14, 8)

import copy
def train_model(model, train_dataset, val_dataset, n_epochs):
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
  criterion = nn.L1Loss(reduction='sum') #reduction='sum'
  history = dict(train=[], val=[])

  best_model_wts = copy.deepcopy(model.state_dict())
  best_loss = 10000.0
  
  for epoch in range(1, n_epochs + 1):
    model = model.train()

    train_losses = []
    for seq_true in train_dataset:
      optimizer.zero_grad()

      seq_true = seq_true
      seq_pred = model(seq_true)

      loss = criterion(seq_pred, seq_true)

      loss.backward()
      optimizer.step()

      train_losses.append(loss.item())

    val_losses = []
    model = model.eval()
    with torch.no_grad():
      for seq_true in val_dataset:

        
        seq_pred = model(seq_true)

        loss = criterion(seq_pred, seq_true)
        val_losses.append(loss.item())

    train_loss = np.mean(train_losses)
    val_loss = np.mean(val_losses)

    history['train'].append(train_loss)
    history['val'].append(val_loss)

    if val_loss < best_loss:
      best_loss = val_loss
      best_model_wts = copy.deepcopy(model.state_dict())

    print(f'Epoch {epoch}: train loss {train_loss} val loss {val_loss}')

  model.load_state_dict(best_model_wts)
  return model.eval(), history



model, history = train_model(
  model, 
  train_df, 
  val_df, 
  n_epochs=15
)



ax = plt.figure().gca()

ax.plot(history['train'])
ax.plot(history['val'])
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'test'])
plt.title('Loss over training epochs')
plt.show()



# Predict
def predict(model, dataset):
  predictions, losses = [], []
  criterion = nn.L1Loss(reduction='sum') #reduction='sum'
  with torch.no_grad():
    model = model.eval()
    for seq_true in dataset:
      
      seq_pred = model(seq_true)

      loss = criterion(seq_pred, seq_true)

      predictions.append(seq_pred.cpu().numpy().flatten())
      losses.append(loss.item())
        
  return predictions, losses



_,pred_healthy=predict(model, train_df)

threshold=np.percentile(pred_healthy,95)

for u in data.unit.unique():
    temp_data=create_sequences_per_engine(data[data.unit==u],5)
    predictions, pred_losses = predict(model, temp_data)
    print('Degrdation for engine ' ,u, 'onsets at time ',np.where(pred_losses>threshold)[0][5])