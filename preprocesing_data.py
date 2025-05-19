# data_preprocessing.py
# Modul untuk pemrosesan data dan persiapan dataset

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def load_data(file_path):
    df = pd.read_csv(file_path)
    
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

def get_ramp_data(df, ramp_id):
    return df[df['ramp'] == ramp_id]

def check_constant_parameters(ramp_data):
    constant_params = {}
    for col in ['Stratigrafi', 'c', 'gamma', 'phi']:
        unique_vals = ramp_data[col].unique()
        constant_params[col] = {
            'is_constant': len(unique_vals) == 1,
            'values': unique_vals
        }
    return constant_params

def prepare_training_data(ramp_data):
    X = ramp_data[['a']]
    y = ramp_data['SF']
    
    # Normalisasi data
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()
    
    # Split data menjadi training dan validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=42
    )
    
    return X_train, X_val, y_train, y_val, scaler_X, scaler_y, X['a'].min(), X['a'].max()