# model.py
# Modul yang berisi definisi dan fungsi-fungsi untuk model ANN

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def Model_ANN(input_shape):
    model = keras.Sequential([
        layers.Dense(30, activation=tf.nn.relu, input_shape=[input_shape]),
        layers.Dense(20, activation=tf.nn.relu),
        layers.Dense(10, activation=tf.nn.relu),
        layers.Dense(1)
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model

def train_model(model, X_train, y_train, X_val=None, y_val=None, epochs=300):
    if X_val is not None and y_val is not None:
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            validation_data=(X_val, y_val),
            verbose=1
        )
    else:
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            verbose=1
        )
    
    return history

def predict_sf(model, a_values_scaled, scaler_y):
    sf_pred_scaled = model.predict(a_values_scaled, verbose=0)
    sf_predictions = scaler_y.inverse_transform(sf_pred_scaled).flatten()
    return sf_predictions

def find_optimal_a(a_range, sf_predictions, target_sf=4.25):
    closest_idx = abs(sf_predictions - target_sf).argmin()
    optimal_a = a_range[closest_idx]
    predicted_sf = sf_predictions[closest_idx]
    
    return optimal_a, predicted_sf