import numpy as np
import pybullet as p
import pybullet_data
import yaml
from pathlib import Path
from tqdm import tqdm


class SimpleFourierDataGenerator:
    """Generate training data using Fourier series torque excitation"""
    
    def __init__(self, config_path="configs/config.yaml", gui=True):
        """Initialize with config"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.robot_config = self.config['robot']
        self.dof = self.robot_config['dof']
        
        # Physics parameters
        self.dt = 0.001  # 1ms simulation timestep (NOT control timestep!)
        self.control_dt = self.config['data']['sampling_time']  # 50ms
        self.substeps = int(self.control_dt / self.dt)
        
        print(f"Simulation dt: {self.dt*1000:.1f}ms")
        print(f"Control dt: {self.control_dt*1000:.1f}ms")
        print(f"Substeps per control: {self.substeps}")
        
        # Initialize PyBullet with GUI
        self.gui = gui
        self.setup_simulation()
        
        # Load robot
        self.robot_id = self.load_robot()
        self.joint_indices = self.get_joint_indices()
        
        # Get limits
        self.q_min = np.array(self.robot_config['joint_limits']['min'])
        self.q_max = np.array(self.robot_config['joint_limits']['max'])
        self.tau_min = np.array(self.robot_config['torque_limits']['min'])
        self.tau_max = np.array(self.robot_config['torque_limits']['max'])
        
        np.random.seed(self.config['seed'])
        
    def setup_simulation(self):
        """Initialize PyBullet"""
        if self.gui:
            self.client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
        else:
            self.client = p.connect(p.DIRECT)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.dt)
        p.setRealTimeSimulation(0)  # Step manually
        
        # Load ground plane
        p.loadURDF("plane.urdf")
        
    def load_robot(self):
        """Load KUKA URDF"""
        urdf_path = self.robot_config['urdf_path']
        if not Path(urdf_path).exists():
            raise FileNotFoundError(f"URDF not found: {urdf_path}")
        
        robot_id = p.loadURDF(
            urdf_path,
            basePosition=[0, 0, 0],
            useFixedBase=True,
            flags=p.URDF_USE_INERTIA_FROM_FILE
        )
        
        print(f"Loaded robot with {p.getNumJoints(robot_id)} joints")
        return robot_id
    
    def get_joint_indices(self):
        """Get revolute joint indices"""
        joint_indices = []
        for i in range(p.getNumJoints(self.robot_id)):
            joint_info = p.getJointInfo(self.robot_id, i)
            joint_type = joint_info[2]
            joint_name = joint_info[1].decode('utf-8')
            
            if joint_type == p.JOINT_REVOLUTE:
                joint_indices.append(i)
                print(f"Joint {i}: {joint_name}")
        
        assert len(joint_indices) == self.dof
        return joint_indices
    
    def reset_robot(self, q_init=None):
        """Reset robot to initial configuration"""
        if q_init is None:
            # Start near zero configuration
            q_init = np.zeros(self.dof)
        
        for idx, pos in zip(self.joint_indices, q_init):
            p.resetJointState(self.robot_id, idx, pos, 0.0)
    
    def get_state(self):
        """Get current joint state"""
        states = p.getJointStates(self.robot_id, self.joint_indices)
        q = np.array([s[0] for s in states])
        dq = np.array([s[1] for s in states])
        return q, dq
    
    def apply_torques(self, torques):
        """Apply torques with proper control mode"""
        # Disable default motor control first
        p.setJointMotorControlArray(
            self.robot_id,
            self.joint_indices,
            p.VELOCITY_CONTROL,
            forces=np.zeros(self.dof)  # Disable built-in PD
        )
        
        # Apply torques
        p.setJointMotorControlArray(
            self.robot_id,
            self.joint_indices,
            p.TORQUE_CONTROL,
            forces=torques
        )
    
    def generate_fourier_torques(self, duration, num_freqs=8):
        """
        Generate rich Fourier series torque trajectories
        
        Args:
            duration: Trajectory duration in seconds
            num_freqs: Number of frequency components per joint
        """
        num_steps = int(duration / self.control_dt)
        t = np.linspace(0, duration, num_steps)
        
        torques = np.zeros((num_steps, self.dof))
        
        print(f"\nGenerating Fourier torques for {duration}s ({num_steps} steps)...")
        
        for joint in range(self.dof):
            # Get torque limit for this joint
            tau_limit = min(abs(self.tau_min[joint]), abs(self.tau_max[joint]))
            
            # Generate multiple frequency components
            for k in range(1, num_freqs + 1):
                # Different fundamental frequency for each joint
                base_freq = 0.05 + 0.05 * joint  # 0.05 to 0.35 Hz
                freq = k * base_freq
                
                # Random amplitude (10-30% of max torque)
                amp = np.random.uniform(0.1, 0.3) * tau_limit
                
                # Random phase
                phase = np.random.uniform(0, 2*np.pi)
                
                # Add component
                torques[:, joint] += amp * np.sin(2*np.pi*freq*t + phase)
            
            print(f"  Joint {joint}: max={torques[:, joint].max():.1f} Nm, "
                  f"min={torques[:, joint].min():.1f} Nm")
        
        # Clip to torque limits (safety)
        torques = np.clip(torques, self.tau_min, self.tau_max)
        
        return torques
    
    def simulate_trajectory(self, torque_trajectory):
        """
        Simulate trajectory with torque input, properly stepping physics
        
        Returns:
            data: Dictionary with states and controls
        """
        num_steps = len(torque_trajectory)
        
        # Storage
        data = {
            'positions': np.zeros((num_steps, self.dof)),
            'velocities': np.zeros((num_steps, self.dof)),
            'torques': np.zeros((num_steps, self.dof)),
            'next_positions': np.zeros((num_steps, self.dof)),
            'next_velocities': np.zeros((num_steps, self.dof))
        }
        
        # Reset to random initial configuration
        q_init = np.random.uniform(self.q_min * 0.3, self.q_max * 0.3)
        self.reset_robot(q_init)
        
        print(f"\nSimulating {num_steps} control steps...")
        for i in tqdm(range(num_steps)):
            # Get current state
            q, dq = self.get_state()
            data['positions'][i] = q
            data['velocities'][i] = dq
            data['torques'][i] = torque_trajectory[i]
            
            # Apply torque for multiple substeps
            for _ in range(self.substeps):
                self.apply_torques(torque_trajectory[i])
                p.stepSimulation()
                
                if self.gui:
                    import time
                    time.sleep(self.dt * 0.5)  # Slow down visualization
            
            # Get next state after all substeps
            q_next, dq_next = self.get_state()
            data['next_positions'][i] = q_next
            data['next_velocities'][i] = dq_next
        
        return data
    
    def add_realistic_noise(self, data):
        """Add sensor noise (optional - probably not needed for sim data)"""
        noise_config = self.config['data']['noise']
        
        noisy_data = {key: val.copy() for key, val in data.items()}
        
        # Only add noise to measurements, not ground truth next states
        noisy_data['positions'] += np.random.normal(
            0, noise_config['position_std'], data['positions'].shape
        )
        noisy_data['velocities'] += np.random.normal(
            0, noise_config['velocity_std'], data['velocities'].shape
        )
        
        return noisy_data
    
    def generate_dataset(self, output_path="data/synthetic_dataset.npz"):
        """Generate complete dataset with multiple trajectories"""
        num_trajectories = self.config['data']['num_trajectories']
        samples_per_traj = self.config['data']['samples_per_trajectory']
        duration = samples_per_traj * self.control_dt
        
        all_data = {
            'positions': [],
            'velocities': [],
            'torques': [],
            'next_positions': [],
            'next_velocities': []
        }
        
        print(f"\n{'='*60}")
        print(f"Generating {num_trajectories} trajectories")
        print(f"Duration per trajectory: {duration}s")
        print(f"{'='*60}")
        
        for traj_idx in range(num_trajectories):
            print(f"\n[Trajectory {traj_idx+1}/{num_trajectories}]")
            
            # Generate Fourier torques
            torques = self.generate_fourier_torques(duration)
            
            # Simulate
            traj_data = self.simulate_trajectory(torques)
            
            # Add noise (optional)
            if self.config['data']['noise']['position_std'] > 0:
                traj_data = self.add_realistic_noise(traj_data)
            
            # Accumulate
            for key in all_data.keys():
                all_data[key].append(traj_data[key])
        
        # Stack all trajectories
        for key in all_data.keys():
            all_data[key] = np.vstack(all_data[key])
        
        total_samples = all_data['positions'].shape[0]
        print(f"\n{'='*60}")
        print(f"Total samples collected: {total_samples}")
        print(f"{'='*60}")
        
        # Train/val/test split
        train_frac = self.config['data']['split']['train']
        val_frac = self.config['data']['split']['val']
        
        train_size = int(train_frac * total_samples)
        val_size = int(val_frac * total_samples)
        
        indices = np.random.permutation(total_samples)
        train_idx = indices[:train_size]
        val_idx = indices[train_size:train_size + val_size]
        test_idx = indices[train_size + val_size:]
        
        # Save
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
        
        print(f"\n✓ Dataset saved to: {output_path}")
        print(f"  Train: {len(train_idx)} samples")
        print(f"  Val:   {len(val_idx)} samples")
        print(f"  Test:  {len(test_idx)} samples")
        
        return output_path
    
    def cleanup(self):
        """Disconnect PyBullet"""
        p.disconnect(self.client)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--gui', action='store_true', help='Show PyBullet GUI')
    parser.add_argument('--config', default='configs/config.yaml', help='Config file')
    args = parser.parse_args()
    
    generator = SimpleFourierDataGenerator(config_path=args.config, gui=args.gui)
    generator.generate_dataset()
    generator.cleanup()
