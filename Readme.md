# Degradation Onset Detection in FD001 Dataset using Transformer Autoencoders

## Overview  
This project aims to detect the onset of degradation in the FD001 dataset using a ML/DL model. The pipeline begins with clustering the CMAPSS sensor data using DBSCAN, which is visualized through t-SNE projection. Since unsupervised models like OCSVM, LSTM Autoencoders, and Transformer Autoencoders are trained exclusively on healthy data, DBSCAN clustering helps estimate the healthy run segments before model training. Once healthy data is identified, various models are trained to learn normal behavior, and degradation onset is detected based on reconstruction error or decision function scores.

## Dataset  
The dataset used is `train_FD001.txt` from the CMAPSS (Commercial Modular Aero-Propulsion System Simulation) dataset. The dataset contains sensor readings from aircraft engines, with each engine's data labeled by unit number and time cycle. The dataset includes the following sensor measurements:  
`s2, s3, s4, s7, s8, s9, s11, s12, s13, s14, s15, s17, s20, s21`

## Preprocessing Steps  
1. **Data Loading**: The dataset is loaded using Pandas, and irrelevant sensors are dropped based on low correlation with RUL (Remaining Useful Life).  
2. **Smoothing**: A moving average with a window size of 30 is applied to smooth the sensor data.  
3. **Normalization**: Min-Max scaling is applied to normalize the sensor data.  
4. **Train-Test Split**:  
   - Healthy data (first 60–70 samples per engine) is split into training and validation sets.  
   - Anomaly data (time > 130 cycles) is used for testing degradation detection (based on literature).

## Transformer Model Architecture  
The model is a Transformer-based Autoencoder with the following components:  
- **Embedding Layer**: Projects input features to a higher-dimensional space.  
- **Positional Encoding**: Adds positional information to the input sequence.  
- **Transformer Encoder**: Processes the input sequence to capture temporal dependencies.  
- **Decoder**: Uses dense layers to reconstruct the input sequence.

## Training  
- **Loss Function**: Mean Squared Error (MSE) between the input and reconstructed sequences.  
- **Optimizer**: Adam with a learning rate of `1e-3`  
- **Batch Size**: 32  
- **Epochs**: 15

## Degradation Detection  
1. **Reconstruction Error**: The trained model reconstructs the entire sequence for each engine.  
2. **Thresholding**: A threshold is set on the reconstruction error (95th percentile of the healthy data) to detect the onset of degradation. The healthy timestamp is chosen based on literature.

## Code Structure  
- **Data Loading and Preprocessing**: Functions to load, smooth, and normalize the data.  
- **Sequence Generation**: Converts the dataset into sequences of fixed length (30 time steps).  
- **Model Definition**: Defines the Transformer Autoencoder architecture.  
- **Training Loop**: Trains the model on healthy data and validates on a separate set.  
- **Visualization**: Plots the training and validation loss curves.

## Results  
- The model achieves low reconstruction error on healthy data and higher error on anomaly data.  
- Degradation onset is identified when the reconstruction error exceeds the predefined threshold.

## References  
- [CMAPSS Dataset – NASA Prognostics Data Repository](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/)  
- Alamr, A.; Artoli, A. *Unsupervised Transformer-Based Anomaly Detection in ECG Signals*. Algorithms 2023, 16, 152. [https://doi.org/10.3390/a16030152](https://doi.org/10.3390/a16030152)  
- Jakubowski, J.; Stanisz, P.; Bobek, S.; Nalepa, G.J. *Anomaly Detection in Asset Degradation Process Using Variational Autoencoder and Explanations*. Sensors 2022, 22, 291. [https://doi.org/10.3390/s22010291](https://doi.org/10.3390/s22010291)
