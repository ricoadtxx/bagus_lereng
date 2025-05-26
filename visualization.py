import matplotlib.pyplot as plt
import numpy as np

def setup_plots(num_ramps):
    plt.figure(figsize=(15, 5 * num_ramps))
    return plt

def plot_predictions(plt_obj, ramp_id, subplot_idx, ramp_data, a_range, sf_predictions, target_sf=4.25):
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
    plt_obj.tight_layout()
    plt_obj.savefig(filename)
    return filename

def display_plot(plt_obj):
    plt_obj.tight_layout()
    plt_obj.show()
    plt_obj.close()  # Close the plot to free memory
    
def plot_combined_predictions(all_predictions, target_sf=1.25, filename='combined_predictions.png'):
    plt.figure(figsize=(10, 6))
    
    for ramp_id, (a_range, sf_pred) in all_predictions.items():
        plt.plot(a_range, sf_pred, label=f'Ramp {ramp_id}')
    
    plt.axhline(y=target_sf, color='g', linestyle='--', label=f'Target SF={target_sf}')
    plt.xlabel('Parameter a')
    plt.ylabel('Safety Factor (SF)')
    plt.title('Gabungan Prediksi ANN: a vs SF')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Plot gabungan disimpan di {filename}")
    plt.show()
    plt.close()  # Close the plot to free memory
