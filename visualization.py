# visualization.py
# Modul untuk visualisasi hasil pelatihan dan prediksi

import matplotlib.pyplot as plt
import numpy as np

def setup_plots(num_ramps):
    """
    Menyiapkan figure dan subplots
    """
    plt.figure(figsize=(15, 5 * num_ramps))
    return plt

def plot_predictions(plt_obj, ramp_id, subplot_idx, ramp_data, a_range, sf_predictions, target_sf=4.25):
    """
    Plot data aktual, prediksi, dan target SF
    """
    plt_obj.subplot(subplot_idx[0], subplot_idx[1], subplot_idx[2])
    plt_obj.scatter(ramp_data['a'], ramp_data['SF'], label=f'Data Aktual (Ramp {ramp_id})')
    plt_obj.plot(a_range, sf_predictions, 'r-', label='Prediksi ANN')
    plt_obj.axhline(y=target_sf, color='g', linestyle='--', label=f'Target SF={target_sf}')
    plt_obj.xlabel('Parameter a')
    plt_obj.ylabel('Safety Factor (SF)')
    plt_obj.title(f'Ramp {ramp_id}: Parameter a vs SF')
    plt_obj.legend()
    plt_obj.grid(True)

def plot_training_history(plt_obj, history, ramp_id, subplot_idx):
    """
    Plot history pelatihan model
    """
    plt_obj.subplot(subplot_idx[0], subplot_idx[1], subplot_idx[2])
    plt_obj.plot(history.history['loss'], label='Training Loss')
    plt_obj.plot(history.history['mae'], label='Training MAE')
    if 'val_loss' in history.history:
        plt_obj.plot(history.history['val_loss'], label='Validation Loss')
        plt_obj.plot(history.history['val_mae'], label='Validation MAE')
    plt_obj.xlabel('Epoch')
    plt_obj.ylabel('Error')
    plt_obj.title(f'Ramp {ramp_id}: History Pelatihan')
    plt_obj.legend()
    plt_obj.grid(True)

def save_plot(plt_obj, filename='sf_prediction_results.png'):
    """
    Menyimpan plot ke file
    """
    plt_obj.tight_layout()
    plt_obj.savefig(filename)
    return filename

def display_plot(plt_obj):
    """
    Menampilkan plot
    """
    plt_obj.tight_layout()
    plt_obj.show()