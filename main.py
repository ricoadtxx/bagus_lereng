import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime

from preprocesing_data import load_data, get_ramp_data, prepare_training_data
from train import Model_ANN, train_model, predict_sf, find_optimal_a
from visualization import setup_plots, plot_predictions, plot_training_history, save_plot, display_plot, plot_combined_predictions

def main():
    results_dir = f"results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    print("Loading dataset...")
    df = load_data('data/dataset.csv')
    
    print("\nDataset Info:")
    print(f"Total rows: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    print("\nSummary statistics:")
    print(df.describe())
    
    ramp_groups = df['ramp'].unique()
    num_ramps = len(ramp_groups)
    print(f"\nNumber of ramps: {num_ramps}")
    
    plt_obj = setup_plots(num_ramps)
    
    optimal_values = {}
    all_predictions = {}
    
    for i, ramp_id in enumerate(ramp_groups):
        print(f"\n{'='*50}")
        print(f"Analyzing Ramp {ramp_id}...")
        print(f"{'='*50}")
        
        ramp_data = get_ramp_data(df, ramp_id)
        
        print("\nPreparing training data...")
        X_train, X_val, y_train, y_val, scaler_X, scaler_y, a_min, a_max = prepare_training_data(ramp_data)
        
        print("\nTraining ANN model...")
        model = Model_ANN(X_train.shape[1])
        history = train_model(model, X_train, y_train, X_val, y_val)
        
        a_range = np.linspace(a_min, a_max, 100)
        a_range_scaled = scaler_X.transform(a_range.reshape(-1, 1))
        
        print("\nMaking predictions...")
        sf_predictions = predict_sf(model, a_range_scaled, scaler_y)
        all_predictions[ramp_id] = (a_range, sf_predictions)
        
        target_sf = 1.25
        optimal_a, predicted_sf = find_optimal_a(a_range, sf_predictions, target_sf)
        
        optimal_values[ramp_id] = {
            'optimal_a': optimal_a,
            'predicted_sf': predicted_sf
        }
        
        print(f"\nNilai Optimal 'a' untuk Target SF={target_sf}: {optimal_a:.2f}")
        print(f"Nilai Prediksi SF untuk a={optimal_a:.2f}: {predicted_sf:.4f}")
        
        plot_predictions(plt_obj, ramp_id, (num_ramps, 2, 2*i+1), 
                       ramp_data, a_range, sf_predictions, target_sf)
        
        plot_training_history(plt_obj, history, ramp_id, (num_ramps, 2, 2*i+2))
        
        model_path = os.path.join(results_dir, f"model_ramp_{ramp_id}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.h5")
        model.save(model_path)
        print(f"Model saved to {model_path}")
    
    plot_path = os.path.join(results_dir, f"results_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png")
    save_plot(plt_obj, plot_path)
    print(f"\nPlot disimpan di {plot_path}")
    
    print("\n" + "="*50)
    print(f"SUMMARY: Nilai Optimal untuk tiap Ramp (Target SF = {target_sf})")
    print("="*50)
    for ramp_id, values in optimal_values.items():
        print(f"Ramp {ramp_id}: a = {values['optimal_a']:.2f}, Predicted SF = {values['predicted_sf']:.4f}")
    
    results_path = os.path.join(results_dir, f"optimal_values_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt")
    combined_plot_path = os.path.join(results_dir, f"combined_predictions_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png")
    
    with open(results_path, 'w') as f:
        f.write(f"Nilai Optimal 'a' untuk tiap Ramp (Target SF = {target_sf})\n")
        f.write("="*50 + "\n")
        for ramp_id, values in optimal_values.items():
            f.write(f"Ramp {ramp_id}: a = {values['optimal_a']:.2f}, Prediksi SF = {values['predicted_sf']:.4f}\n")
    print(f"\nHasil disimpan di {results_path}")
    
    display_plot(plt_obj)

    plot_combined_predictions(all_predictions, target_sf, combined_plot_path)
    
if __name__ == "__main__":
    main()