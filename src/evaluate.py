"""
Evaluation script for comparing Baseline DNN vs Transformer MPC
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from pathlib import Path
import sys
from tqdm import tqdm
import argparse

sys.path.append(str(Path(__file__).parent.parent))
from models.baseline_dnn import create_baseline_model
from models.transformer_predictor import create_transformer_model
from mpc_controller import LearnedDynamicsMPC, load_trained_model


class MPCEvaluator:
    """Evaluate MPC performance"""
    
    def __init__(self, config):
        self.config = config
        self.dof = config['robot']['dof']
        self.dt = config['mpc']['sampling_time']
        
        # Setup plotting
        sns.set_style("whitegrid")
        self.fig_dir = Path("figures")
        self.fig_dir.mkdir(exist_ok=True)
    
    def generate_reference_trajectory(self, traj_type, duration, q_init):
        """Generate reference trajectory for testing"""
        num_steps = int(duration / self.dt)
        t = np.linspace(0, duration, num_steps)
        
        q_min = np.array(self.config['robot']['joint_limits']['min'])
        q_max = np.array(self.config['robot']['joint_limits']['max'])
        q_center = (q_min + q_max) / 2
        
        if traj_type == 'point_stabilization':
            # Step changes every 2 seconds
            q_ref = np.zeros((self.dof, num_steps))
            step_duration = int(2.0 / self.dt)
            
            for i in range(0, num_steps, step_duration):
                q_target = np.random.uniform(q_min * 0.7, q_max * 0.7)
                end_idx = min(i + step_duration, num_steps)
                q_ref[:, i:end_idx] = q_target[:, None]
        
        elif traj_type == 'circular':
            radius = 0.3
            omega = 2 * np.pi / duration
            q_ref = np.tile(q_center[:, None], (1, num_steps))
            q_ref[0, :] = q_center[0] + radius * np.cos(omega * t)
            q_ref[1, :] = q_center[1] + radius * np.sin(omega * t)
        
        elif traj_type == 'figure_eight':
            omega = 2 * np.pi / duration
            scale = 0.3
            q_ref = np.tile(q_center[:, None], (1, num_steps))
            q_ref[0, :] = q_center[0] + scale * np.sin(omega * t)
            q_ref[1, :] = q_center[1] + scale * np.sin(omega * t) * np.cos(omega * t)
        
        elif traj_type == 'sinusoidal':
            q_ref = np.tile(q_center[:, None], (1, num_steps))
            for j in range(self.dof):
                freq = 0.5 + 0.5 * j / self.dof
                amplitude = 0.2 * (q_max[j] - q_min[j])
                q_ref[j, :] = q_center[j] + amplitude * np.sin(2 * np.pi * freq * t)
        
        else:
            raise ValueError(f"Unknown trajectory type: {traj_type}")
        
        return q_ref
    
    def simulate_mpc(self, mpc, q_init, q_ref_trajectory):
        """Simulate MPC control"""
        num_steps = q_ref_trajectory.shape[1]
        
        # Storage
        q_history = np.zeros((self.dof, num_steps))
        dq_history = np.zeros((self.dof, num_steps))
        tau_history = np.zeros((self.dof, num_steps))
        
        # Initialize
        q = q_init.copy()
        dq = np.zeros(self.dof)
        
        # If transformer, reset history
        if hasattr(mpc, 'reset_history'):
            mpc.reset_history()
        
        for i in tqdm(range(num_steps), desc="Simulating MPC"):
            # Store state
            q_history[:, i] = q
            dq_history[:, i] = dq
            
            # Get reference for prediction horizon
            start_idx = i
            end_idx = min(i + mpc.N + 1, num_steps)
            horizon_length = end_idx - start_idx
            
            if horizon_length < mpc.N + 1:
                # Pad with last value
                q_ref = np.hstack([
                    q_ref_trajectory[:, start_idx:end_idx],
                    np.tile(q_ref_trajectory[:, -1:], (1, mpc.N + 1 - horizon_length))
                ])
            else:
                q_ref = q_ref_trajectory[:, start_idx:end_idx]
            
            # Compute control
            try:
                tau = mpc.step(q, dq, q_ref)
                tau_history[:, i] = tau
            except Exception as e:
                print(f"MPC failed at step {i}: {e}")
                tau = np.zeros(self.dof)
                tau_history[:, i] = tau
            
            # Update history for transformer
            if hasattr(mpc, 'update_history'):
                mpc.update_history(q, dq, tau)
            
            # Simulate dynamics using learned model
            q_next, dq_next = mpc.predict_next_state_casadi(q, dq, tau)
            
            # Update state
            q = q_next
            dq = dq_next
        
        return q_history, dq_history, tau_history
    
    def compute_metrics(self, q_actual, q_ref):
        """Compute tracking metrics"""
        error = q_actual - q_ref
        
        metrics = {
            'mse': np.mean(error ** 2),
            'rmse': np.sqrt(np.mean(error ** 2)),
            'mae': np.mean(np.abs(error)),
            'max_error': np.max(np.abs(error))
        }
        
        return metrics
    
    def plot_trajectory_tracking(self, q_actual, q_ref, tau, title, save_name):
        """Plot trajectory tracking results"""
        num_steps = q_actual.shape[1]
        t = np.arange(num_steps) * self.dt
        
        fig, axes = plt.subplots(self.dof + 1, 1, figsize=(12, 2.5 * (self.dof + 1)))
        
        # Plot each joint
        for i in range(self.dof):
            axes[i].plot(t, q_ref[i, :], 'r--', label='Reference', linewidth=2)
            axes[i].plot(t, q_actual[i, :], 'b-', label='Actual', linewidth=1.5)
            axes[i].set_ylabel(f'Joint {i+1} [rad]')
            axes[i].legend()
            axes[i].grid(True)
        
        # Plot torques
        for i in range(self.dof):
            axes[-1].plot(t, tau[i, :], label=f'Joint {i+1}')
        axes[-1].set_xlabel('Time [s]')
        axes[-1].set_ylabel('Torque [Nm]')
        axes[-1].legend(ncol=self.dof)
        axes[-1].grid(True)
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.fig_dir / f"{save_name}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_comparison(self, results_baseline, results_transformer, scenario_name):
        """Compare baseline vs transformer performance"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot joint 1 tracking
        t = np.arange(results_baseline['q'].shape[1]) * self.dt
        axes[0, 0].plot(t, results_baseline['q_ref'][0, :], 'k--', 
                       label='Reference', linewidth=2)
        axes[0, 0].plot(t, results_baseline['q'][0, :], 'b-', 
                       label='Baseline DNN', linewidth=1.5, alpha=0.7)
        axes[0, 0].plot(t, results_transformer['q'][0, :], 'r-', 
                       label='Transformer', linewidth=1.5, alpha=0.7)
        axes[0, 0].set_xlabel('Time [s]')
        axes[0, 0].set_ylabel('Joint 1 Position [rad]')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        axes[0, 0].set_title('Joint 1 Tracking')
        
        # Plot tracking error
        error_baseline = np.linalg.norm(
            results_baseline['q'] - results_baseline['q_ref'], axis=0
        )
        error_transformer = np.linalg.norm(
            results_transformer['q'] - results_transformer['q_ref'], axis=0
        )
        
        axes[0, 1].plot(t, error_baseline, 'b-', label='Baseline DNN', linewidth=1.5)
        axes[0, 1].plot(t, error_transformer, 'r-', label='Transformer', linewidth=1.5)
        axes[0, 1].set_xlabel('Time [s]')
        axes[0, 1].set_ylabel('Tracking Error [rad]')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        axes[0, 1].set_title('Total Tracking Error')
        axes[0, 1].set_yscale('log')
        
        # Metrics comparison
        metrics_baseline = results_baseline['metrics']
        metrics_transformer = results_transformer['metrics']
        
        metric_names = list(metrics_baseline.keys())
        baseline_values = [metrics_baseline[m] for m in metric_names]
        transformer_values = [metrics_transformer[m] for m in metric_names]
        
        x = np.arange(len(metric_names))
        width = 0.35
        
        axes[1, 0].bar(x - width/2, baseline_values, width, label='Baseline DNN')
        axes[1, 0].bar(x + width/2, transformer_values, width, label='Transformer')
        axes[1, 0].set_ylabel('Error')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels([m.upper() for m in metric_names])
        axes[1, 0].legend()
        axes[1, 0].set_title('Performance Metrics')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, axis='y')
        
        # Control effort comparison
        control_baseline = np.sum(results_baseline['tau'] ** 2, axis=0)
        control_transformer = np.sum(results_transformer['tau'] ** 2, axis=0)
        
        axes[1, 1].plot(t, control_baseline, 'b-', label='Baseline DNN', linewidth=1.5)
        axes[1, 1].plot(t, control_transformer, 'r-', label='Transformer', linewidth=1.5)
        axes[1, 1].set_xlabel('Time [s]')
        axes[1, 1].set_ylabel('Control Effort [Nm²]')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        axes[1, 1].set_title('Control Effort')
        
        plt.suptitle(f'Comparison: {scenario_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.fig_dir / f"comparison_{scenario_name}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()


def evaluate_scenario(evaluator, mpc_baseline, mpc_transformer, scenario):
    """Evaluate single scenario"""
    print(f"\n{'='*60}")
    print(f"Evaluating: {scenario['name']}")
    print(f"{'='*60}")
    
    # Generate reference trajectory
    q_init = np.zeros(evaluator.dof)
    q_ref = evaluator.generate_reference_trajectory(
        scenario.get('trajectory_type', scenario['name']),
        scenario['duration'],
        q_init
    )
    
    # Simulate baseline
    print("\nBaseline DNN:")
    q_baseline, dq_baseline, tau_baseline = evaluator.simulate_mpc(
        mpc_baseline, q_init, q_ref
    )
    metrics_baseline = evaluator.compute_metrics(q_baseline, q_ref)
    
    print("\nBaseline metrics:")
    for k, v in metrics_baseline.items():
        print(f"  {k.upper()}: {v:.6f}")
    
    # Simulate transformer
    print("\nTransformer:")
    q_transformer, dq_transformer, tau_transformer = evaluator.simulate_mpc(
        mpc_transformer, q_init, q_ref
    )
    metrics_transformer = evaluator.compute_metrics(q_transformer, q_ref)
    
    print("\nTransformer metrics:")
    for k, v in metrics_transformer.items():
        print(f"  {k.upper()}: {v:.6f}")
    
    # Compute improvement
    improvement = {}
    for k in metrics_baseline.keys():
        improvement[k] = (metrics_baseline[k] - metrics_transformer[k]) / metrics_baseline[k] * 100
    
    print("\nImprovement (%):")
    for k, v in improvement.items():
        print(f"  {k.upper()}: {v:+.2f}%")
    
    # Plot individual results
    evaluator.plot_trajectory_tracking(
        q_baseline, q_ref, tau_baseline,
        f"Baseline DNN - {scenario['name']}",
        f"baseline_{scenario['name']}"
    )
    
    evaluator.plot_trajectory_tracking(
        q_transformer, q_ref, tau_transformer,
        f"Transformer - {scenario['name']}",
        f"transformer_{scenario['name']}"
    )
    
    # Plot comparison
    results = {
        'baseline': {
            'q': q_baseline,
            'dq': dq_baseline,
            'tau': tau_baseline,
            'q_ref': q_ref,
            'metrics': metrics_baseline
        },
        'transformer': {
            'q': q_transformer,
            'dq': dq_transformer,
            'tau': tau_transformer,
            'q_ref': q_ref,
            'metrics': metrics_transformer
        }
    }
    
    evaluator.plot_comparison(
        results['baseline'],
        results['transformer'],
        scenario['name']
    )
    
    return results, improvement


def main(args):
    """Main evaluation function"""
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load trained models
    print("Loading trained models...")
    model_baseline = load_trained_model('baseline', config)
    model_transformer = load_trained_model('transformer', config)
    
    # Create MPC controllers
    print("\nCreating MPC controllers...")
    mpc_baseline = LearnedDynamicsMPC(model_baseline, config, 'baseline')
    mpc_transformer = LearnedDynamicsMPC(model_transformer, config, 'transformer')
    
    # Create evaluator
    evaluator = MPCEvaluator(config)
    
    # Evaluate scenarios
    all_results = {}
    all_improvements = {}
    
    for scenario in config['evaluation']['test_scenarios']:
        results, improvement = evaluate_scenario(
            evaluator, mpc_baseline, mpc_transformer, scenario
        )
        all_results[scenario['name']] = results
        all_improvements[scenario['name']] = improvement
    
    # Print summary
    print(f"\n{'='*60}")
    print("OVERALL SUMMARY")
    print(f"{'='*60}")
    
    for scenario_name, improvement in all_improvements.items():
        print(f"\n{scenario_name}:")
        for k, v in improvement.items():
            print(f"  {k.upper()} improvement: {v:+.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate MPC performance')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    
    args = parser.parse_args()
    main(args)
