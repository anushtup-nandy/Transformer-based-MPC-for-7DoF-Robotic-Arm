"""
Simplified Evaluation Script with PyBullet GUI
Now supports comparing any two models: baseline / transformer / lstm
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

# make repo root importable
sys.path.append(str(Path(__file__).parent.parent))

# model factories
from models.baseline_dnn import create_baseline_model
from models.transformer_predictor import create_transformer_model
from models.lstm_predictor import create_lstm_model

# MPC utilities from the repo (we only reuse the MPC class; we load models ourselves)
from mpc_controller import LearnedDynamicsMPC


# ----------------------- Simulator (unchanged) -----------------------
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
            self.set_joint_positions(q_history[:, i])

            self.update_text(
                f"{model_name}\n"
                f"Step: {i+1}/{num_steps}\n"
                f"Time: {i*dt:.2f}s"
            )

            p.stepSimulation()

            elapsed = time.time() - start_time
            sleep_time = (dt / speed) - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

            keys = p.getKeyboardEvents()
            if ord('q') in keys and keys[ord('q')] & p.KEY_WAS_TRIGGERED:
                print("Visualization skipped by user")
                break

    def cleanup(self):
        """Disconnect PyBullet"""
        p.disconnect(self.client)


# ----------------------- Evaluator -----------------------
class MPCEvaluator:
    """Evaluate MPC performance"""

    def __init__(self, config, use_gui=False):
        self.config = config
        self.dof = config['robot']['dof']
        self.dt = config['mpc']['sampling_time']
        self.use_gui = use_gui

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
            if mpc.model_type in ('transformer', 'lstm'):
                for _ in range(mpc.history_length):
                    mpc.position_history.append(q.copy())
                    mpc.velocity_history.append(dq.copy())
                    mpc.torque_history.append(np.zeros(self.dof))

        for i in tqdm(range(num_steps), desc=f"Simulating {model_name}"):
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

    def plot_comparison(self, results_a, results_b, scenario_name, name_a, name_b):
        """Compare two models"""
        fig = plt.figure(figsize=(18, 12))
        t = np.arange(results_a['q'].shape[1]) * self.dt

        # Plot all 7 joints
        for joint_idx in range(self.dof):
            ax = plt.subplot(3, 3, joint_idx + 1)
            ax.plot(t, results_a['q_ref'][joint_idx, :], 'k--',
                    label='Reference', linewidth=2, alpha=0.5)
            ax.plot(t, results_a['q'][joint_idx, :], 'b-',
                    label=name_a, linewidth=1.5, alpha=0.7)
            ax.plot(t, results_b['q'][joint_idx, :], 'r-',
                    label=name_b, linewidth=1.5, alpha=0.7)
            ax.set_xlabel('Time [s]')
            ax.set_ylabel(f'Joint {joint_idx+1} [rad]')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_title(f'Joint {joint_idx+1} Tracking')

        # Total tracking error (subplot 8)
        ax = plt.subplot(3, 3, 8)
        error_a = np.linalg.norm(results_a['q'] - results_a['q_ref'], axis=0)
        error_b = np.linalg.norm(results_b['q'] - results_b['q_ref'], axis=0)
        ax.plot(t, error_a, 'b-', label=name_a, linewidth=1.5)
        ax.plot(t, error_b, 'r-', label=name_b, linewidth=1.5)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Total Error [rad]')
        ax.legend()
        ax.grid(True)
        ax.set_title('Total Tracking Error')
        ax.set_yscale('log')

        # Metrics comparison (subplot 9)
        ax = plt.subplot(3, 3, 9)
        metrics_a = results_a['metrics']
        metrics_b = results_b['metrics']

        metric_names = list(metrics_a.keys())
        a_values = [metrics_a[m] for m in metric_names]
        b_values = [metrics_b[m] for m in metric_names]

        x = np.arange(len(metric_names))
        width = 0.35

        ax.bar(x - width/2, a_values, width, label=name_a)
        ax.bar(x + width/2, b_values, width, label=name_b)
        ax.set_ylabel('Error')
        ax.set_xticks(x)
        ax.set_xticklabels([m.upper() for m in metric_names], fontsize=8)
        ax.legend()
        ax.set_title('Performance Metrics')
        ax.set_yscale('log')
        ax.grid(True, axis='y', alpha=0.3)

        plt.suptitle(f'Comparison: {scenario_name}', fontsize=16, fontweight='bold')
        out = self.fig_dir / f"comparison_{scenario_name}_{name_a}_vs_{name_b}.png"
        plt.tight_layout()
        plt.savefig(out, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved plot: {out}")

    def cleanup(self):
        if self.sim is not None:
            self.sim.cleanup()


# ----------------------- Helpers -----------------------
def build_model(model_type: str, config, hidden_dim=128, num_layers=2, dropout=0.1):
    if model_type == 'baseline':
        return create_baseline_model(config), 'baseline'
    elif model_type == 'transformer':
        return create_transformer_model(config), 'transformer'
    elif model_type == 'lstm':
        return create_lstm_model(config, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout), 'lstm'
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def load_checkpoint_if_any(model, ckpt_path: str):
    """Load either a raw state_dict or the Trainer checkpoint dict."""
    if ckpt_path is None:
        return
    state = torch.load(ckpt_path, map_location='cpu')
    if isinstance(state, dict) and 'model_state_dict' in state:
        model.load_state_dict(state['model_state_dict'])
    else:
        model.load_state_dict(state)


def default_ckpt_path(model_type: str) -> Path:
    return Path("models") / "trained" / model_type / "best.pth"


# ----------------------- Main -----------------------
def main(args):
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Build and load Model A
    model_a, tag_a = build_model(args.model_a, config,
                                 hidden_dim=args.hidden_dim,
                                 num_layers=args.num_layers,
                                 dropout=args.dropout)
    ckpt_a = Path(args.ckpt_a) if args.ckpt_a else default_ckpt_path(tag_a)
    if ckpt_a.exists():
        print(f"Loading checkpoint for {tag_a}: {ckpt_a}")
        load_checkpoint_if_any(model_a, str(ckpt_a))
    else:
        print(f"Warning: checkpoint not found for {tag_a}: {ckpt_a} (using random init)")

    # Build and load Model B
    model_b, tag_b = build_model(args.model_b, config,
                                 hidden_dim=args.hidden_dim,
                                 num_layers=args.num_layers,
                                 dropout=args.dropout)
    ckpt_b = Path(args.ckpt_b) if args.ckpt_b else default_ckpt_path(tag_b)
    if ckpt_b.exists():
        print(f"Loading checkpoint for {tag_b}: {ckpt_b}")
        load_checkpoint_if_any(model_b, str(ckpt_b))
    else:
        print(f"Warning: checkpoint not found for {tag_b}: {ckpt_b} (using random init)")

    # Create MPC controllers
    print("\nCreating MPC controllers...")
    mpc_a = LearnedDynamicsMPC(model_a, config, tag_a)
    mpc_b = LearnedDynamicsMPC(model_b, config, tag_b)

    evaluator = MPCEvaluator(config, use_gui=args.gui)

    # Pick scenario
    scenarios = config['evaluation']['test_scenarios']
    scenario = next((s for s in scenarios if s['name'] == args.scenario), None)
    if scenario is None:
        print(f"Error: Scenario '{args.scenario}' not found!")
        print(f"Available: {[s['name'] for s in scenarios]}")
        return

    print(f"\n{'='*60}")
    print(f"Evaluating: {scenario['name']} ({tag_a} vs {tag_b})")
    print(f"{'='*60}")

    # Reference trajectory
    q_init = np.zeros(evaluator.dof)
    q_ref = evaluator.generate_reference_trajectory(
        scenario.get('trajectory_type', scenario['name']),
        scenario['duration'],
        q_init
    )

    # Model A
    print(f"\n[1/2] {tag_a.upper()}:")
    q_a, dq_a, tau_a = evaluator.simulate_mpc(mpc_a, q_init, q_ref, tag_a.upper())
    metrics_a = evaluator.compute_metrics(q_a, q_ref)
    print(f"\n{tag_a.upper()} metrics:")
    for k, v in metrics_a.items():
        print(f"  {k.upper()}: {v:.6f}")

    # Model B
    print(f"\n[2/2] {tag_b.upper()}:")
    q_b, dq_b, tau_b = evaluator.simulate_mpc(mpc_b, q_init, q_ref, tag_b.upper())
    metrics_b = evaluator.compute_metrics(q_b, q_ref)
    print(f"\n{tag_b.upper()} metrics:")
    for k, v in metrics_b.items():
        print(f"  {k.UPPER() if hasattr(k,'UPPER') else k.upper()}: {v:.6f}")

    # Optional GUI playback
    if args.gui:
        if evaluator.sim is None:
            evaluator.sim = RobotSimulator(config, gui=True)
        evaluator.sim.visualize_trajectory(q_a, tag_a.upper(),
                                           dt=evaluator.dt, speed=args.speed)
        evaluator.sim.visualize_trajectory(q_b, tag_b.upper(),
                                           dt=evaluator.dt, speed=args.speed)

    # Improvement print
    print("\n" + "="*60)
    print(f"IMPROVEMENT of {tag_b} over {tag_a} (%)")
    print("="*60)
    for k in metrics_a.keys():
        base = metrics_a[k]
        new = metrics_b[k]
        improvement = (base - new) / (base + 1e-12) * 100.0
        symbol = "✓" if improvement > 0 else "✗"
        print(f"  {symbol} {k.upper()}: {improvement:+.2f}%")

    # Plot
    results_a = {'q': q_a, 'dq': dq_a, 'tau': tau_a, 'q_ref': q_ref, 'metrics': metrics_a}
    results_b = {'q': q_b, 'dq': dq_b, 'tau': tau_b, 'q_ref': q_ref, 'metrics': metrics_b}
    evaluator.plot_comparison(results_a, results_b, scenario['name'], tag_a.upper(), tag_b.upper())

    evaluator.cleanup()
    print(f"\n{'='*60}\nEVALUATION COMPLETE\n{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate MPC performance (compare two models)')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    parser.add_argument('--scenario', type=str, default='point_stabilization',
                        choices=['point_stabilization', 'trajectory_tracking', 'complex_tracking'],
                        help='Which scenario to evaluate')
    parser.add_argument('--gui', action='store_true',
                        help='Show PyBullet GUI visualization')
    parser.add_argument('--speed', type=float, default=1.0,
                        help='Playback speed for visualization (1.0 = real-time)')

    # NEW: choose any two models to compare
    parser.add_argument('--model_a', choices=['baseline', 'transformer', 'lstm'], default='baseline')
    parser.add_argument('--model_b', choices=['baseline', 'transformer', 'lstm'], default='lstm')

    # Optional explicit checkpoints (otherwise auto: models/trained/<type>/best.pth)
    parser.add_argument('--ckpt_a', type=str, default=None)
    parser.add_argument('--ckpt_b', type=str, default=None)

    # LSTM hyperparams (ignored for baseline/transformer)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout',   type=float, default=0.1)

    args = parser.parse_args()
    main(args)
