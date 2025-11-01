"""
Simplified Evaluation Script - Select specific scenarios
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
        
        sns.set_style("whitegrid")
        self.fig_dir = Path("figures")
        self.fig_dir.mkdir(exist_ok=True)
    
    def generate_reference_trajectory(self, traj_type, duration, q_init):
        """Generate reference trajectory"""
        num_steps = int(duration / self.dt)
        t = np.linspace(0, duration, num_steps)
        
        q_min = np.array(self.config['robot']['joint_limits']['min'])
        q_max = np.array(self.config['robot']['joint_limits']['max'])
        q_center = (q_min + q_max) / 2
        
        if traj_type == 'point_stabilization':
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
        
        else:
            raise ValueError(f"Unknown trajectory type: {traj_type}")
        
        return q_ref
    
    def simulate_mpc(self, mpc, q_init, q_ref_trajectory):
        """Simulate MPC control"""
        num_steps = q_ref_trajectory.shape[1]
        
        q_history = np.zeros((self.dof, num_steps))
        dq_history = np.zeros((self.dof, num_steps))
        tau_history = np.zeros((self.dof, num_steps))
        
        q = q_init.copy()
        dq = np.zeros(self.dof)
        
        if hasattr(mpc, 'reset_history'):
            mpc.reset_history()
        
        for i in tqdm(range(num_steps), desc="Simulating MPC"):
            q_history[:, i] = q
            dq_history[:, i] = dq
            
            start_idx = i
            end_idx = min(i + mpc.N + 1, num_steps)
            horizon_length = end_idx - start_idx
            
            if horizon_length < mpc.N + 1:
                q_ref = np.hstack([
                    q_ref_trajectory[:, start_idx:end_idx],
                    np.tile(q_ref_trajectory[:, -1:], (1, mpc.N + 1 - horizon_length))
                ])
            else:
                q_ref = q_ref_trajectory[:, start_idx:end_idx]
            
            try:
                tau = mpc.step(q, dq, q_ref)
                tau_history[:, i] = tau
            except Exception as e:
                print(f"MPC failed at step {i}: {e}")
                tau = np.zeros(self.dof)
                tau_history[:, i] = tau
            
            if hasattr(mpc, 'update_history'):
                mpc.update_history(q, dq, tau)
            
            q_next, dq_next = mpc.predict_next_state_numpy(q, dq, tau)
            
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
    
    def plot_comparison(self, results_baseline, results_transformer, scenario_name):
        """Compare baseline vs transformer"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        t = np.arange(results_baseline['q'].shape[1]) * self.dt
        
        # Joint 1 tracking
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
        
        # Tracking error
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
        
        # Control effort
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
        print(f"Saved plot: {self.fig_dir / f'comparison_{scenario_name}.png'}")


def main(args):
    """Main evaluation function"""
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load models
    print("Loading trained models...")
    model_baseline = load_trained_model('baseline', config)
    model_transformer = load_trained_model('transformer', config)
    
    # Create MPC controllers
    print("\nCreating MPC controllers...")
    mpc_baseline = LearnedDynamicsMPC(model_baseline, config, 'baseline')
    mpc_transformer = LearnedDynamicsMPC(model_transformer, config, 'transformer')
    
    evaluator = MPCEvaluator(config)
    
    # Find scenario
    scenarios = config['evaluation']['test_scenarios']
    scenario = None
    for s in scenarios:
        if s['name'] == args.scenario:
            scenario = s
            break
    
    if scenario is None:
        print(f"Error: Scenario '{args.scenario}' not found!")
        print(f"Available scenarios: {[s['name'] for s in scenarios]}")
        return
    
    print(f"\n{'='*60}")
    print(f"Evaluating: {scenario['name']}")
    print(f"{'='*60}")
    
    # Generate reference
    q_init = np.zeros(evaluator.dof)
    q_ref = evaluator.generate_reference_trajectory(
        scenario.get('trajectory_type', scenario['name']),
        scenario['duration'],
        q_init
    )
    
    # Baseline
    print("\nBaseline DNN:")
    q_baseline, dq_baseline, tau_baseline = evaluator.simulate_mpc(
        mpc_baseline, q_init, q_ref
    )
    metrics_baseline = evaluator.compute_metrics(q_baseline, q_ref)
    
    print("\nBaseline metrics:")
    for k, v in metrics_baseline.items():
        print(f"  {k.upper()}: {v:.6f}")
    
    # Transformer
    print("\nTransformer:")
    q_transformer, dq_transformer, tau_transformer = evaluator.simulate_mpc(
        mpc_transformer, q_init, q_ref
    )
    metrics_transformer = evaluator.compute_metrics(q_transformer, q_ref)
    
    print("\nTransformer metrics:")
    for k, v in metrics_transformer.items():
        print(f"  {k.upper()}: {v:.6f}")
    
    # Improvement
    print("\nImprovement (%):")
    for k in metrics_baseline.keys():
        improvement = (metrics_baseline[k] - metrics_transformer[k]) / metrics_baseline[k] * 100
        print(f"  {k.upper()}: {improvement:+.2f}%")
    
    # Plot
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
    
    evaluator.plot_comparison(results['baseline'], results['transformer'], scenario['name'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate MPC performance')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    parser.add_argument('--scenario', type=str, default='point_stabilization',
                        choices=['point_stabilization', 'trajectory_tracking', 'complex_tracking'],
                        help='Which scenario to evaluate')
    
    args = parser.parse_args()
    main(args)
