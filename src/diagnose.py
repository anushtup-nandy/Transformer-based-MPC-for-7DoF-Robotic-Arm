"""
Diagnostic Script - Check data quality and model predictions
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import yaml
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from models.baseline_dnn import create_baseline_model
from models.transformer_predictor import create_transformer_model


def check_data_quality(data_path):
    """Check if training data makes sense"""
    print("\n" + "="*60)
    print("DATA QUALITY CHECK")
    print("="*60)
    
    data = np.load(data_path)
    
    # Check training data
    train_pos = data['train_positions']
    train_vel = data['train_velocities']
    train_tau = data['train_torques']
    train_next_pos = data['train_next_positions']
    train_next_vel = data['train_next_velocities']
    
    print(f"\nTraining set size: {len(train_pos)}")
    print(f"Shape: pos={train_pos.shape}, vel={train_vel.shape}, tau={train_tau.shape}")
    
    # Check delta statistics
    delta_pos = train_next_pos - train_pos
    delta_vel = train_next_vel - train_vel
    
    print(f"\nDelta Position Statistics (should be small for 50ms steps):")
    print(f"  Mean: {np.mean(np.abs(delta_pos), axis=0)}")
    print(f"  Std:  {np.std(delta_pos, axis=0)}")
    print(f"  Max:  {np.max(np.abs(delta_pos), axis=0)}")
    
    print(f"\nDelta Velocity Statistics:")
    print(f"  Mean: {np.mean(np.abs(delta_vel), axis=0)}")
    print(f"  Std:  {np.std(delta_vel, axis=0)}")
    print(f"  Max:  {np.max(np.abs(delta_vel), axis=0)}")
    
    # Check if dynamics are reasonable (delta should correlate with velocity)
    print(f"\nDynamics Check:")
    for i in range(7):
        # delta_pos should be approximately vel * dt
        dt = 0.05
        expected_delta = train_vel[:, i] * dt
        actual_delta = delta_pos[:, i]
        
        correlation = np.corrcoef(expected_delta, actual_delta)[0, 1]
        print(f"  Joint {i}: pos_delta vs vel*dt correlation = {correlation:.4f}")
    
    # Check torque range
    print(f"\nTorque Statistics:")
    print(f"  Mean: {np.mean(np.abs(train_tau), axis=0)}")
    print(f"  Std:  {np.std(train_tau, axis=0)}")
    print(f"  Max:  {np.max(np.abs(train_tau), axis=0)}")
    
    # Plot sample trajectory
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    
    sample_len = 200
    t = np.arange(sample_len) * 0.05
    
    axes[0].plot(t, train_pos[:sample_len, :])
    axes[0].set_ylabel('Position [rad]')
    axes[0].set_title('Sample Trajectory - Positions')
    axes[0].grid(True)
    
    axes[1].plot(t, train_vel[:sample_len, :])
    axes[1].set_ylabel('Velocity [rad/s]')
    axes[1].set_title('Sample Trajectory - Velocities')
    axes[1].grid(True)
    
    axes[2].plot(t, train_tau[:sample_len, :])
    axes[2].set_ylabel('Torque [Nm]')
    axes[2].set_xlabel('Time [s]')
    axes[2].set_title('Sample Trajectory - Torques')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig('figures/data_quality.png', dpi=300)
    print(f"\nSaved data quality plot: figures/data_quality.png")
    plt.close()
    
    return data


def check_model_predictions(model, model_type, data, config):
    """Check if model predictions are reasonable"""
    print("\n" + "="*60)
    print(f"{model_type.upper()} MODEL PREDICTION CHECK")
    print("="*60)
    
    model.eval()
    device = torch.device('cpu')
    model.to(device)
    
    # Check normalization stats
    print(f"\nModel Normalization Statistics:")
    print(f"  pos_mean: {model.pos_mean.cpu().numpy()}")
    print(f"  pos_std:  {model.pos_std.cpu().numpy()}")
    print(f"  vel_mean: {model.vel_mean.cpu().numpy()}")
    print(f"  vel_std:  {model.vel_std.cpu().numpy()}")
    print(f"  tau_mean: {model.tau_mean.cpu().numpy()}")
    print(f"  tau_std:  {model.tau_std.cpu().numpy()}")
    
    # Test on validation data
    val_pos = data['val_positions']
    val_vel = data['val_velocities']
    val_tau = data['val_torques']
    val_next_pos = data['val_next_positions']
    val_next_vel = data['val_next_velocities']
    
    # Take first 100 samples
    n_test = 100
    
    if model_type == 'baseline':
        # Baseline takes single timestep
        if val_pos.ndim == 3:
            test_pos = val_pos[:n_test, -1, :]
            test_vel = val_vel[:n_test, -1, :]
            test_tau = val_tau[:n_test, -1, :]
        else:
            test_pos = val_pos[:n_test]
            test_vel = val_vel[:n_test]
            test_tau = val_tau[:n_test]
    else:
        # Transformer needs sequences
        test_pos = val_pos[:n_test]
        test_vel = val_vel[:n_test]
        test_tau = val_tau[:n_test]
    
    test_next_pos = val_next_pos[:n_test]
    test_next_vel = val_next_vel[:n_test]
    
    # Predict
    with torch.no_grad():
        test_pos_t = torch.FloatTensor(test_pos).to(device)
        test_vel_t = torch.FloatTensor(test_vel).to(device)
        test_tau_t = torch.FloatTensor(test_tau).to(device)
        
        pred_state = model(test_pos_t, test_vel_t, test_tau_t)
        pred_pos = pred_state[:, :7].cpu().numpy()
        pred_vel = pred_state[:, 7:].cpu().numpy()
    
    # Compute errors
    pos_error = pred_pos - test_next_pos
    vel_error = pred_vel - test_next_vel
    
    print(f"\nPrediction Errors (on {n_test} validation samples):")
    print(f"\nPosition Error:")
    print(f"  Mean: {np.mean(np.abs(pos_error), axis=0)}")
    print(f"  Std:  {np.std(pos_error, axis=0)}")
    print(f"  RMSE: {np.sqrt(np.mean(pos_error**2, axis=0))}")
    print(f"  Max:  {np.max(np.abs(pos_error), axis=0)}")
    
    print(f"\nVelocity Error:")
    print(f"  Mean: {np.mean(np.abs(vel_error), axis=0)}")
    print(f"  Std:  {np.std(vel_error, axis=0)}")
    print(f"  RMSE: {np.sqrt(np.mean(vel_error**2, axis=0))}")
    print(f"  Max:  {np.max(np.abs(vel_error), axis=0)}")
    
    # Overall metrics
    pos_mse = np.mean(pos_error**2)
    vel_mse = np.mean(vel_error**2)
    total_mse = pos_mse + vel_mse
    
    print(f"\nOverall Metrics:")
    print(f"  Position MSE: {pos_mse:.6f}")
    print(f"  Velocity MSE: {vel_mse:.6f}")
    print(f"  Total MSE: {total_mse:.6f}")
    print(f"  Total RMSE: {np.sqrt(total_mse):.6f}")
    
    # Plot predictions vs ground truth
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Position predictions for joint 0
    axes[0].plot(test_next_pos[:50, 0], 'b-', label='Ground Truth', linewidth=2)
    axes[0].plot(pred_pos[:50, 0], 'r--', label='Predicted', linewidth=2)
    axes[0].set_ylabel('Joint 0 Position [rad]')
    axes[0].set_title(f'{model_type} - Position Predictions')
    axes[0].legend()
    axes[0].grid(True)
    
    # Velocity predictions for joint 0
    axes[1].plot(test_next_vel[:50, 0], 'b-', label='Ground Truth', linewidth=2)
    axes[1].plot(pred_vel[:50, 0], 'r--', label='Predicted', linewidth=2)
    axes[1].set_ylabel('Joint 0 Velocity [rad/s]')
    axes[1].set_xlabel('Sample Index')
    axes[1].set_title(f'{model_type} - Velocity Predictions')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(f'figures/predictions_{model_type}.png', dpi=300)
    print(f"\nSaved prediction plot: figures/predictions_{model_type}.png")
    plt.close()


def main():
    """Run diagnostics"""
    # Load config
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Check data
    data = check_data_quality('data/synthetic_dataset.npz')
    
    # Load and check baseline model
    try:
        checkpoint = torch.load('models/trained/baseline/best.pth', map_location='cpu')
        model_baseline = create_baseline_model(config)
        model_baseline.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"\nBaseline model loaded from epoch {checkpoint['epoch']}")
        print(f"Best validation loss: {checkpoint['best_val_loss']:.6f}")
        
        check_model_predictions(model_baseline, 'baseline', data, config)
    except Exception as e:
        print(f"\nError loading baseline model: {e}")
    
    # Load and check transformer model
    try:
        checkpoint = torch.load('models/trained/transformer/best.pth', map_location='cpu')
        model_transformer = create_transformer_model(config)
        model_transformer.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"\nTransformer model loaded from epoch {checkpoint['epoch']}")
        print(f"Best validation loss: {checkpoint['best_val_loss']:.6f}")
        
        check_model_predictions(model_transformer, 'transformer', data, config)
    except Exception as e:
        print(f"\nError loading transformer model: {e}")
    
    print("\n" + "="*60)
    print("DIAGNOSIS COMPLETE")
    print("="*60)
    print("\nKey Issues to Check:")
    print("1. Is delta_pos ≈ vel * 0.05? (Should be high correlation)")
    print("2. Are normalization stats non-zero in models?")
    print("3. Is prediction error reasonable (< 0.1 rad)?")
    print("4. Does training loss match validation predictions?")


if __name__ == "__main__":
    main()
