"""
Synthetic Data Generator for KUKA LBR4 using PyBullet
Generates realistic trajectories with noise for training the Transformer-based MPC
"""

import numpy as np
import pybullet as p
import pybullet_data
import yaml
from pathlib import Path
from tqdm import tqdm
import time


class KukaDataGenerator:
    """Generate synthetic training data for KUKA LBR4 manipulator"""
    
    def __init__(self, config_path="configs/config.yaml"):
        """Initialize the data generator with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.robot_config = self.config['robot']
        self.data_config = self.config['data']
        self.dof = self.robot_config['dof']
        self.dt = self.data_config['sampling_time']
        
        # Initialize PyBullet
        self.setup_simulation()
        
        # Load robot
        self.robot_id = self.load_robot()
        
        # Get joint information
        self.joint_indices = self.get_joint_indices()
        
        # Set random seed
        np.random.seed(self.config['seed'])
        
    def setup_simulation(self):
        """Initialize PyBullet simulation"""
        self.client = p.connect(p.DIRECT)  # Headless mode
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.dt)
        
    def load_robot(self):
        """Load KUKA robot URDF"""
        urdf_path = self.robot_config['urdf_path']
        if not Path(urdf_path).exists():
            raise FileNotFoundError(f"URDF not found at {urdf_path}")
        
        robot_id = p.loadURDF(
            urdf_path,
            basePosition=[0, 0, 0],
            useFixedBase=True,
            flags=p.URDF_USE_INERTIA_FROM_FILE
        )
        return robot_id
    
    def get_joint_indices(self):
        """Get indices of revolute joints"""
        joint_indices = []
        for i in range(p.getNumJoints(self.robot_id)):
            joint_info = p.getJointInfo(self.robot_id, i)
            if joint_info[2] == p.JOINT_REVOLUTE:  # Only revolute joints
                joint_indices.append(i)
        
        assert len(joint_indices) == self.dof, \
            f"Expected {self.dof} joints, found {len(joint_indices)}"
        
        return joint_indices
    
    def get_joint_limits(self):
        """Get joint position limits"""
        q_min = np.array(self.robot_config['joint_limits']['min'])
        q_max = np.array(self.robot_config['joint_limits']['max'])
        return q_min, q_max
    
    def get_torque_limits(self):
        """Get torque limits"""
        tau_min = np.array(self.robot_config['torque_limits']['min'])
        tau_max = np.array(self.robot_config['torque_limits']['max'])
        return tau_min, tau_max
    
    def set_joint_positions(self, positions):
        """Set joint positions"""
        for idx, pos in zip(self.joint_indices, positions):
            p.resetJointState(self.robot_id, idx, pos, 0.0)
    
    def get_joint_states(self):
        """Get current joint positions and velocities"""
        states = p.getJointStates(self.robot_id, self.joint_indices)
        positions = np.array([s[0] for s in states])
        velocities = np.array([s[1] for s in states])
        return positions, velocities
    
    def apply_torques(self, torques):
        """Apply torques to joints"""
        p.setJointMotorControlArray(
            self.robot_id,
            self.joint_indices,
            p.TORQUE_CONTROL,
            forces=torques
        )
    
    def generate_pick_and_place_trajectory(self, num_samples):
        """Generate pick and place trajectory"""
        q_min, q_max = self.get_joint_limits()
        
        # Generate random start and end configurations
        q_start = np.random.uniform(q_min * 0.7, q_max * 0.7)
        q_end = np.random.uniform(q_min * 0.7, q_max * 0.7)
        
        # Generate smooth trajectory using minimum jerk
        t = np.linspace(0, 1, num_samples)
        s = 10 * t**3 - 15 * t**4 + 6 * t**5  # Minimum jerk interpolation
        
        trajectory = q_start[None, :] + s[:, None] * (q_end - q_start)[None, :]
        return trajectory
    
    def generate_circular_trajectory(self, num_samples):
        """Generate circular trajectory in joint space"""
        q_min, q_max = self.get_joint_limits()
        q_center = (q_min + q_max) / 2
        
        # Use first 3 joints for circular motion
        radius = 0.3
        t = np.linspace(0, 2 * np.pi, num_samples)
        
        trajectory = np.tile(q_center, (num_samples, 1))
        trajectory[:, 0] = q_center[0] + radius * np.cos(t)
        trajectory[:, 1] = q_center[1] + radius * np.sin(t)
        
        return trajectory
    
    def generate_figure_eight_trajectory(self, num_samples):
        """Generate figure-8 trajectory"""
        q_min, q_max = self.get_joint_limits()
        q_center = (q_min + q_max) / 2
        
        t = np.linspace(0, 2 * np.pi, num_samples)
        scale = 0.3
        
        trajectory = np.tile(q_center, (num_samples, 1))
        trajectory[:, 0] = q_center[0] + scale * np.sin(t)
        trajectory[:, 1] = q_center[1] + scale * np.sin(t) * np.cos(t)
        
        return trajectory
    
    def generate_sinusoidal_trajectory(self, num_samples):
        """Generate sinusoidal motion"""
        q_min, q_max = self.get_joint_limits()
        q_center = (q_min + q_max) / 2
        
        t = np.linspace(0, 4 * np.pi, num_samples)
        trajectory = np.tile(q_center, (num_samples, 1))
        
        # Apply different frequencies to each joint
        for i in range(self.dof):
            freq = 0.5 + 0.5 * i / self.dof
            amplitude = 0.2 * (q_max[i] - q_min[i])
            trajectory[:, i] = q_center[i] + amplitude * np.sin(freq * t)
        
        return trajectory
    
    def generate_trajectory(self, traj_type, num_samples):
        """Generate trajectory based on type"""
        if traj_type == "pick_and_place":
            return self.generate_pick_and_place_trajectory(num_samples)
        elif traj_type == "circular":
            return self.generate_circular_trajectory(num_samples)
        elif traj_type == "figure_eight":
            return self.generate_figure_eight_trajectory(num_samples)
        elif traj_type == "sinusoidal":
            return self.generate_sinusoidal_trajectory(num_samples)
        else:
            raise ValueError(f"Unknown trajectory type: {traj_type}")
    
    def simulate_trajectory(self, desired_trajectory):
        """Simulate trajectory and collect data with PD control"""
        num_samples = len(desired_trajectory)
        
        # Storage
        data = {
            'positions': np.zeros((num_samples, self.dof)),
            'velocities': np.zeros((num_samples, self.dof)),
            'torques': np.zeros((num_samples, self.dof)),
            'next_positions': np.zeros((num_samples, self.dof)),
            'next_velocities': np.zeros((num_samples, self.dof))
        }
        
        # PD control gains
        kp = np.array([500, 500, 300, 300, 200, 100, 100])
        kd = np.array([50, 50, 30, 30, 20, 10, 10])
        
        # Initialize at start position
        self.set_joint_positions(desired_trajectory[0])
        
        for i in range(num_samples):
            # Get current state
            q_current, dq_current = self.get_joint_states()
            
            # PD control
            q_desired = desired_trajectory[i]
            dq_desired = np.zeros(self.dof)
            if i < num_samples - 1:
                dq_desired = (desired_trajectory[i+1] - q_desired) / self.dt
            
            error = q_desired - q_current
            error_dot = dq_desired - dq_current
            torques = kp * error + kd * error_dot
            
            # Clip torques
            tau_min, tau_max = self.get_torque_limits()
            torques = np.clip(torques, tau_min, tau_max)
            
            # Store current state
            data['positions'][i] = q_current
            data['velocities'][i] = dq_current
            data['torques'][i] = torques
            
            # Apply control and step simulation
            self.apply_torques(torques)
            p.stepSimulation()
            
            # Get next state
            q_next, dq_next = self.get_joint_states()
            data['next_positions'][i] = q_next
            data['next_velocities'][i] = dq_next
        
        return data
    
    def add_noise(self, data):
        """Add realistic sensor noise to data"""
        noise_config = self.data_config['noise']
        
        noisy_data = data.copy()
        
        # Add Gaussian noise
        noisy_data['positions'] += np.random.normal(
            0, noise_config['position_std'], data['positions'].shape
        )
        noisy_data['velocities'] += np.random.normal(
            0, noise_config['velocity_std'], data['velocities'].shape
        )
        noisy_data['torques'] += np.random.normal(
            0, noise_config['torque_std'], data['torques'].shape
        )
        noisy_data['next_positions'] += np.random.normal(
            0, noise_config['position_std'], data['next_positions'].shape
        )
        noisy_data['next_velocities'] += np.random.normal(
            0, noise_config['velocity_std'], data['next_velocities'].shape
        )
        
        return noisy_data
    
    def generate_dataset(self, output_path="data/synthetic_dataset.npz"):
        """Generate complete dataset"""
        num_trajectories = self.data_config['num_trajectories']
        samples_per_traj = self.data_config['samples_per_trajectory']
        
        # Calculate trajectory distribution
        traj_types = []
        weights = []
        for traj_info in self.data_config['trajectories']:
            traj_types.append(traj_info['type'])
            weights.append(traj_info['weight'])
        
        weights = np.array(weights) / sum(weights)
        
        # Generate trajectories
        all_data = {
            'positions': [],
            'velocities': [],
            'torques': [],
            'next_positions': [],
            'next_velocities': []
        }
        
        print("Generating synthetic dataset...")
        for i in tqdm(range(num_trajectories)):
            # Select trajectory type
            traj_type = np.random.choice(traj_types, p=weights)
            
            # Generate desired trajectory
            desired_traj = self.generate_trajectory(traj_type, samples_per_traj)
            
            # Simulate and collect data
            traj_data = self.simulate_trajectory(desired_traj)
            
            # Add noise
            noisy_data = self.add_noise(traj_data)
            
            # Accumulate
            for key in all_data.keys():
                all_data[key].append(noisy_data[key])
        
        # Convert to arrays
        for key in all_data.keys():
            all_data[key] = np.vstack(all_data[key])
        
        # Split into train/val/test
        total_samples = all_data['positions'].shape[0]
        train_size = int(self.data_config['split']['train'] * total_samples)
        val_size = int(self.data_config['split']['val'] * total_samples)
        
        indices = np.random.permutation(total_samples)
        train_idx = indices[:train_size]
        val_idx = indices[train_size:train_size + val_size]
        test_idx = indices[train_size + val_size:]
        
        # Save dataset
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            output_path,
            train_positions=all_data['positions'][train_idx],
            train_velocities=all_data['velocities'][train_idx],
            train_torques=all_data['torques'][train_idx],
            train_next_positions=all_data['next_positions'][train_idx],
            train_next_velocities=all_data['next_velocities'][train_idx],
            val_positions=all_data['positions'][val_idx],
            val_velocities=all_data['velocities'][val_idx],
            val_torques=all_data['torques'][val_idx],
            val_next_positions=all_data['next_positions'][val_idx],
            val_next_velocities=all_data['next_velocities'][val_idx],
            test_positions=all_data['positions'][test_idx],
            test_velocities=all_data['velocities'][test_idx],
            test_torques=all_data['torques'][test_idx],
            test_next_positions=all_data['next_positions'][test_idx],
            test_next_velocities=all_data['next_velocities'][test_idx]
        )
        
        print(f"\nDataset saved to {output_path}")
        print(f"Train samples: {len(train_idx)}")
        print(f"Val samples: {len(val_idx)}")
        print(f"Test samples: {len(test_idx)}")
        
        return output_path
    
    def cleanup(self):
        """Cleanup PyBullet"""
        p.disconnect(self.client)


if __name__ == "__main__":
    generator = KukaDataGenerator()
    generator.generate_dataset()
    generator.cleanup()
