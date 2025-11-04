"""
Simplified Evaluation Script with PyBullet GUI - Select specific scenarios
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
import pybullet as p
import pybullet_data
import time

sys.path.append(str(Path(__file__).parent.parent))
from models.baseline_dnn import create_baseline_model
from models.transformer_predictor import create_transformer_model
from mpc_controller import LearnedDynamicsMPC, load_trained_model


class RobotSimulator:
    """PyBullet simulator for visualization"""
    
    def __init__(self, config, gui=True):
        self.config = config
        self.dof = config['robot']['dof']
        
        # Connect to PyBullet
        if gui:
            self.client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
            p.resetDebugVisualizerCamera(
                cameraDistance=2.0,
                cameraYaw=45,
                cameraPitch=-30,
                cameraTargetPosition=[0, 0, 0.5]
            )
        else:
            self.client = p.connect(p.DIRECT)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        # Load ground plane
        p.loadURDF("plane.urdf")
        
        # Load robot
        urdf_path = config['robot']['urdf_path']
        self.robot_id = p.loadURDF(
            urdf_path,
            basePosition=[0, 0, 1],
            useFixedBase=True,
            flags=p.URDF_USE_INERTIA_FROM_FILE
        )
        
        # Get joint indices
        self.joint_indices = []
        for i in range(p.getNumJoints(self.robot_id)):
            joint_info = p.getJointInfo(self.robot_id, i)
            if joint_info[2] == p.JOINT_REVOLUTE:
                self.joint_indices.append(i)
        
        assert len(self.joint_indices) == self.dof
        
        # Add text display for info
        self.text_id = None
        
        print(f"✓ PyBullet simulator initialized (GUI={'ON' if gui else 'OFF'})")
    
    def set_joint_positions(self, q):
        """Set robot joint positions"""
        for idx, pos in zip(self.joint_indices, q):
            p.resetJointState(self.robot_id, idx, pos, 0.0)
    
    def update_text(self, text):
        """Update text display in GUI"""
        if self.text_id is not None:
            p.removeUserDebugItem(self.text_id)
        
        self.text_id = p.addUserDebugText(
            text,
            textPosition=[0, 0, 2],
            textColorRGB=[1, 0, 0],
            textSize=1.5
        )
    
    def visualize_trajectory(self, q_history, model_name, dt=0.05, speed=1.0):
        """Visualize trajectory in PyBullet"""
        num_steps = q_history.shape[1]
        
        print(f"\nVisualizing {model_name} trajectory...")
        print("Press 'q' to skip visualization")
        
        for i in range(num_steps):
            start_time = time.time()
            
            # Update robot pose
            self.set_joint_positions(q_history[:, i])
            
            # Update info text
            self.update_text(
                f"{model_name}\n"
                f"Step: {i+1}/{num_steps}\n"
                f"Time: {i*dt:.2f}s"
            )
            
            # Step simulation (just for visualization, not physics)
            p.stepSimulation()
            
            # Sleep to maintain real-time visualization
            elapsed = time.time() - start_time
            sleep_time = (dt / speed) - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            
            # Check for quit
            keys = p.getKeyboardEvents()
            if ord('q') in keys and keys[ord('q')] & p.KEY_WAS_TRIGGERED:
                print("Visualization skipped by user")
                break
    
    def cleanup(self):
        """Disconnect PyBullet"""
        p.disconnect(self.client)


class MPCEvaluator:
    """Evaluate MPC performance"""
    
    def __init__(self, config, use_gui=False):
        self.config = config
        self.dof = config['robot']['dof']
        self.dt = config['mpc']['sampling_time']
        self.use_gui = use_gui
        
        # Initialize simulator if GUI enabled
        if use_gui:
            self.sim = RobotSimulator(config, gui=True)
        else:
            self.sim = None
        
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
    
    def simulate_mpc(self, mpc, q_init, q_ref_trajectory, model_name="MPC"):
        """Simulate MPC control"""
        num_steps = q_ref_trajectory.shape[1]
        
        q_history = np.zeros((self.dof, num_steps))
        dq_history = np.zeros((self.dof, num_steps))
        tau_history = np.zeros((self.dof, num_steps))
        
        q = q_init.copy()
        dq = np.zeros(self.dof)
        
        if hasattr(mpc, 'reset_history'):
            mpc.reset_history()
            if mpc.model_type == 'transformer':
                for _ in range(mpc.history_length):
                    mpc.position_history.append(q.copy())
                    mpc.velocity_history.append(dq.copy())
                    mpc.torque_history.append(np.zeros(self.dof))
        
        for i in tqdm(range(num_steps), desc=f"Simulating {model_name}"):
            q_history[:, i] = q
            dq_history[:, i] = dq

            # if self.sim is not None:
            #     self.sim.set_joint_positions(q)
            #     self.sim.update_text(
            #         f"{model_name} (Computing...)\n"
            #         f"Step: {i+1}/{num_steps}\n"
            #         f"Time: {i*self.dt:.2f}s"
            #     )
            #     p.stepSimulation()
            
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
            
            # if hasattr(mpc, 'update_history'):
            #     mpc.update_history(q, dq, tau)
            
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
        # Create figure with more subplots for all joints
        fig = plt.figure(figsize=(18, 12))
        
        t = np.arange(results_baseline['q'].shape[1]) * self.dt
        
        # Plot all 7 joints
        for joint_idx in range(self.dof):
            ax = plt.subplot(3, 3, joint_idx + 1)
            ax.plot(t, results_baseline['q_ref'][joint_idx, :], 'k--', 
                   label='Reference', linewidth=2, alpha=0.5)
            ax.plot(t, results_baseline['q'][joint_idx, :], 'b-', 
                   label='Baseline DNN', linewidth=1.5, alpha=0.7)
            ax.plot(t, results_transformer['q'][joint_idx, :], 'r-', 
                   label='Transformer', linewidth=1.5, alpha=0.7)
            ax.set_xlabel('Time [s]')
            ax.set_ylabel(f'Joint {joint_idx+1} [rad]')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_title(f'Joint {joint_idx+1} Tracking')
        
        # Total tracking error (subplot 8)
        ax = plt.subplot(3, 3, 8)
        error_baseline = np.linalg.norm(
            results_baseline['q'] - results_baseline['q_ref'], axis=0
        )
        error_transformer = np.linalg.norm(
            results_transformer['q'] - results_transformer['q_ref'], axis=0
        )
        ax.plot(t, error_baseline, 'b-', label='Baseline DNN', linewidth=1.5)
        ax.plot(t, error_transformer, 'r-', label='Transformer', linewidth=1.5)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Total Error [rad]')
        ax.legend()
        ax.grid(True)
        ax.set_title('Total Tracking Error')
        ax.set_yscale('log')
        
        # Metrics comparison (subplot 9)
        ax = plt.subplot(3, 3, 9)
        metrics_baseline = results_baseline['metrics']
        metrics_transformer = results_transformer['metrics']
        
        metric_names = list(metrics_baseline.keys())
        baseline_values = [metrics_baseline[m] for m in metric_names]
        transformer_values = [metrics_transformer[m] for m in metric_names]
        
        x = np.arange(len(metric_names))
        width = 0.35
        
        ax.bar(x - width/2, baseline_values, width, label='Baseline DNN')
        ax.bar(x + width/2, transformer_values, width, label='Transformer')
        ax.set_ylabel('Error')
        ax.set_xticks(x)
        ax.set_xticklabels([m.upper() for m in metric_names], fontsize=8)
        ax.legend()
        ax.set_title('Performance Metrics')
        ax.set_yscale('log')
        ax.grid(True, axis='y', alpha=0.3)
        
        plt.suptitle(f'Comparison: {scenario_name}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.fig_dir / f"comparison_{scenario_name}_all_joints.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved plot: {self.fig_dir / f'comparison_{scenario_name}_all_joints.png'}") 
    
    def cleanup(self):
        """Cleanup resources"""
        if self.sim is not None:
            self.sim.cleanup()


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
    
    evaluator = MPCEvaluator(config, use_gui=False)
    
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

    # Transformer
    print("\n[1/2] Transformer:")
    q_transformer, dq_transformer, tau_transformer = evaluator.simulate_mpc(
        mpc_transformer, q_init, q_ref, "Transformer"
    )
    metrics_transformer = evaluator.compute_metrics(q_transformer, q_ref)
    
    print("\nTransformer metrics:")
    for k, v in metrics_transformer.items():
        print(f"  {k.upper()}: {v:.6f}")
    
    # Baseline
    print("\n[2/2] Baseline DNN:")
    q_baseline, dq_baseline, tau_baseline = evaluator.simulate_mpc(
        mpc_baseline, q_init, q_ref, "Baseline DNN"
    )
    metrics_baseline = evaluator.compute_metrics(q_baseline, q_ref)
    
    print("\nBaseline metrics:")
    for k, v in metrics_baseline.items():
        print(f"  {k.upper()}: {v:.6f}")
    
    if args.gui:
        evaluator.sim = RobotSimulator(config, gui=True)
        evaluator.sim.visualize_trajectory(q_transformer, "Transformer", 
                                          dt=evaluator.dt, speed=args.speed)
        evaluator.sim.visualize_trajectory(q_baseline, "Baseline DNN",
                                          dt=evaluator.dt, speed=args.speed) 
    
    # Improvement
    print("\n" + "="*60)
    print("IMPROVEMENT (%)")
    print("="*60)
    for k in metrics_baseline.keys():
        improvement = (metrics_baseline[k] - metrics_transformer[k]) / metrics_baseline[k] * 100
        symbol = "✓" if improvement > 0 else "✗"
        print(f"  {symbol} {k.upper()}: {improvement:+.2f}%")
    
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
    
    evaluator.plot_comparison(results['baseline'], results['transformer'], scenario['name'])
    
    # Cleanup
    evaluator.cleanup()
    
    print(f"\n{'='*60}")
    print("EVALUATION COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate MPC performance with visualization')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    parser.add_argument('--scenario', type=str, default='point_stabilization',
                        choices=['point_stabilization', 'trajectory_tracking', 'complex_tracking'],
                        help='Which scenario to evaluate')
    parser.add_argument('--gui', action='store_true',
                        help='Show PyBullet GUI visualization')
    parser.add_argument('--speed', type=float, default=1.0,
                        help='Playback speed for visualization (1.0 = real-time)')
    
    args = parser.parse_args()
    main(args)
