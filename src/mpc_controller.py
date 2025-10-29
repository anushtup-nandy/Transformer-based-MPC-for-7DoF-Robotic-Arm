"""
Model Predictive Controller using learned dynamics models
Implements data-driven MPC with CasADi and IPOPT
"""

import casadi as ca
import numpy as np
import torch
import yaml
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))


class LearnedDynamicsMPC:
    """
    MPC controller using learned neural network dynamics
    Compatible with both DNN and Transformer models
    """
    
    def __init__(self, model, config, model_type='transformer'):
        """
        Args:
            model: Trained PyTorch model (DNN or Transformer)
            config: Configuration dictionary
            model_type: 'baseline' or 'transformer'
        """
        self.model = model
        self.model.eval()  # Set to evaluation mode
        self.config = config
        self.model_type = model_type
        
        # Robot parameters
        robot_config = config['robot']
        self.dof = robot_config['dof']
        self.q_min = np.array(robot_config['joint_limits']['min'])
        self.q_max = np.array(robot_config['joint_limits']['max'])
        self.tau_min = np.array(robot_config['torque_limits']['min'])
        self.tau_max = np.array(robot_config['torque_limits']['max'])
        self.dq_min = np.array(robot_config['velocity_limits']['min'])
        self.dq_max = np.array(robot_config['velocity_limits']['max'])
        
        # MPC parameters
        mpc_config = config['mpc']
        self.N = mpc_config['prediction_horizon']
        self.dt = mpc_config['sampling_time']
        self.W1 = mpc_config['weights']['state']
        self.W2 = mpc_config['weights']['control']
        
        # For transformer: history management
        if model_type == 'transformer':
            self.history_length = config['transformer']['history_length']
            self.reset_history()
        else:
            self.history_length = 1 
        
        # Build MPC problem
        self.build_mpc()
        
        print(f"MPC initialized with horizon N={self.N}")
    
    def reset_history(self):
        """Reset history buffer for transformer"""
        self.position_history = []
        self.velocity_history = []
        self.torque_history = []
    
    def update_history(self, q, dq, tau):
        """Update history for transformer"""
        self.position_history.append(q.copy())
        self.velocity_history.append(dq.copy())
        self.torque_history.append(tau.copy())
        
        if len(self.position_history) > self.history_length:
            self.position_history.pop(0)
            self.velocity_history.pop(0)
            self.torque_history.pop(0)
    
    def predict_next_state_casadi(self, q, dq, tau):
        """
        Predict next state using learned model
        This wraps the PyTorch model for use in CasADi
        
        Args:
            q: (7,) joint positions
            dq: (7,) joint velocities  
            tau: (7,) torques
            
        Returns:
            q_next: (7,) next positions
            dq_next: (7,) next velocities
        """
        # Convert to numpy for PyTorch
        if isinstance(q, ca.MX) or isinstance(q, ca.SX):
            # This will be called symbolically, return symbolic output
            # We'll handle this differently in the MPC setup
            return q, dq  # Placeholder
        
        # For numerical evaluation
        with torch.no_grad():
            if self.model_type == 'baseline':
                # Single step prediction
                q_tensor = torch.FloatTensor(q).unsqueeze(0)
                dq_tensor = torch.FloatTensor(dq).unsqueeze(0)
                tau_tensor = torch.FloatTensor(tau).unsqueeze(0)
                
                next_state = self.model(q_tensor, dq_tensor, tau_tensor)
                q_next = next_state[0, :7].numpy()
                dq_next = next_state[0, 7:].numpy()
            else:
                # Transformer: use history
                if len(self.position_history) < self.history_length:
                    # Pad with current state
                    while len(self.position_history) < self.history_length:
                        self.position_history.append(q.copy())
                        self.velocity_history.append(dq.copy())
                        self.torque_history.append(tau.copy())
                
                # Stack history
                hist_q = torch.FloatTensor(np.array(self.position_history)).unsqueeze(0)
                hist_dq = torch.FloatTensor(np.array(self.velocity_history)).unsqueeze(0)
                hist_tau = torch.FloatTensor(np.array(self.torque_history)).unsqueeze(0)
                
                next_state = self.model(hist_q, hist_dq, hist_tau)
                q_next = next_state[0, :7].numpy()
                dq_next = next_state[0, 7:].numpy()
        
        return q_next, dq_next
    
    def export_model_to_casadi(self):
        """
        Export PyTorch model to CasADi function
        This creates a symbolic function that can be used in optimization
        """
        # Create symbolic inputs
        q_sym = ca.SX.sym('q', self.dof)
        dq_sym = ca.SX.sym('dq', self.dof)
        tau_sym = ca.SX.sym('tau', self.dof)
        
        # For baseline DNN, we can create a simpler forward pass
        # For transformer, this is more complex due to history
        
        # Extract model parameters as numpy arrays
        params = []
        for param in self.model.parameters():
            params.append(param.detach().cpu().numpy())
        
        # Build CasADi equivalent (simplified for baseline)
        if self.model_type == 'baseline':
            # Concatenate inputs
            x = ca.vertcat(q_sym, dq_sym, tau_sym)
            
            # First layer
            W1 = params[0]
            b1 = params[1]
            h1 = ca.mtimes(W1, x) + b1
            h1 = ca.fmax(0, h1)  # ReLU
            
            # Second layer  
            W2 = params[2]
            b2 = params[3]
            h2 = ca.mtimes(W2, h1) + b2
            h2 = ca.fmax(0, h2)  # ReLU
            
            # Output layer
            W3 = params[4]
            b3 = params[5]
            output = ca.mtimes(W3, h2) + b3
            
            q_next = output[:self.dof]
            dq_next = output[self.dof:]
            
            # Create CasADi function
            dynamics_func = ca.Function(
                'learned_dynamics',
                [q_sym, dq_sym, tau_sym],
                [q_next, dq_next],
                ['q', 'dq', 'tau'],
                ['q_next', 'dq_next']
            )
            
            return dynamics_func
        else:
            # For transformer, we use external callback
            # This is more complex and we'll use a different approach
            return None
    
    def build_mpc(self):
        """Build the MPC optimization problem using CasADi"""
        # Use Opti stack for easier formulation
        self.opti = ca.Opti()
        
        # Decision variables
        # States: [q, dq] for each timestep
        self.X = self.opti.variable(self.dof * 2, self.N + 1)
        
        # Controls: tau for each timestep
        self.U = self.opti.variable(self.dof, self.N)
        
        # Parameters
        self.x0 = self.opti.parameter(self.dof * 2)  # Initial state
        self.x_ref = self.opti.parameter(self.dof * 2, self.N + 1)  # Reference trajectory
        
        # Cost function
        cost = 0
        
        for k in range(self.N):
            # State at k
            q_k = self.X[:self.dof, k]
            dq_k = self.X[self.dof:, k]
            tau_k = self.U[:, k]
            
            # Reference at k
            q_ref_k = self.x_ref[:self.dof, k]
            dq_ref_k = self.x_ref[self.dof:, k]
            
            # Tracking error
            e_q = q_k - q_ref_k
            e_dq = dq_k - dq_ref_k
            
            # Stage cost
            cost += self.W1 * (ca.dot(e_q, e_q) + ca.dot(e_dq, e_dq))
            
            # Control effort
            if k > 0:
                delta_tau = tau_k - self.U[:, k-1]
                cost += self.W2 * ca.dot(delta_tau, delta_tau)
            
            # Dynamics constraints (simplified - will use callback)
            # For now, use a simple forward Euler approximation
            # In practice, this would use the learned model
            q_next = q_k + self.dt * dq_k
            dq_next = dq_k + self.dt * tau_k / 100.0  # Simplified
            
            self.opti.subject_to(self.X[:self.dof, k+1] == q_next)
            self.opti.subject_to(self.X[self.dof:, k+1] == dq_next)
        
        # Terminal cost
        e_q_final = self.X[:self.dof, self.N] - self.x_ref[:self.dof, self.N]
        e_dq_final = self.X[self.dof:, self.N] - self.x_ref[self.dof:, self.N]
        cost += self.W1 * (ca.dot(e_q_final, e_q_final) + ca.dot(e_dq_final, e_dq_final))
        
        # Objective
        self.opti.minimize(cost)
        
        # Initial condition constraint
        self.opti.subject_to(self.X[:, 0] == self.x0)
        
        # State constraints
        if self.config['mpc']['apply_joint_constraints']:
            for k in range(self.N + 1):
                self.opti.subject_to(self.opti.bounded(
                    self.q_min, self.X[:self.dof, k], self.q_max
                ))
        
        if self.config['mpc']['apply_velocity_constraints']:
            for k in range(self.N + 1):
                self.opti.subject_to(self.opti.bounded(
                    self.dq_min, self.X[self.dof:, k], self.dq_max
                ))
        
        # Control constraints
        if self.config['mpc']['apply_torque_constraints']:
            for k in range(self.N):
                self.opti.subject_to(self.opti.bounded(
                    self.tau_min, self.U[:, k], self.tau_max
                ))
        
        # Solver options
        solver_opts = self.config['mpc']['solver']['options']
        p_opts = {"expand": True}
        s_opts = {
            "max_iter": solver_opts['max_iter'],
            "tol": solver_opts['tol'],
            "acceptable_tol": solver_opts['acceptable_tol'],
            "print_level": solver_opts['print_level']
        }
        
        self.opti.solver('ipopt', p_opts, s_opts)
        
        print("MPC problem built successfully")
    
    def solve(self, x_current, x_reference):
        """
        Solve MPC optimization problem
        
        Args:
            x_current: (14,) current state [q, dq]
            x_reference: (14, N+1) reference trajectory
            
        Returns:
            u_opt: (7,) optimal control (first step)
            x_pred: (14, N+1) predicted state trajectory
        """
        # Set initial condition
        self.opti.set_value(self.x0, x_current)
        
        # Set reference trajectory
        self.opti.set_value(self.x_ref, x_reference)
        
        # Solve
        try:
            sol = self.opti.solve()
            
            # Extract solution
            x_opt = sol.value(self.X)
            u_opt = sol.value(self.U)
            
            # Return first control action
            return u_opt[:, 0], x_opt
        
        except RuntimeError as e:
            print(f"MPC solve failed: {e}")
            # Return zero torque as fallback
            return np.zeros(self.dof), None
    
    def step(self, q_current, dq_current, q_ref, dq_ref=None):
        """
        Single MPC control step
        
        Args:
            q_current: (7,) current joint positions
            dq_current: (7,) current joint velocities
            q_ref: (7,) or (7, N+1) reference positions
            dq_ref: (7,) or (7, N+1) reference velocities (optional)
            
        Returns:
            tau: (7,) optimal torque
        """
        # Prepare current state
        x_current = np.concatenate([q_current, dq_current])
        
        # Prepare reference trajectory
        if q_ref.ndim == 1:
            # Single target - repeat for horizon
            q_ref = np.tile(q_ref[:, None], (1, self.N + 1))
            if dq_ref is None:
                dq_ref = np.zeros((self.dof, self.N + 1))
            elif dq_ref.ndim == 1:
                dq_ref = np.tile(dq_ref[:, None], (1, self.N + 1))
        else:
            if dq_ref is None:
                dq_ref = np.zeros((self.dof, self.N + 1))
        
        x_reference = np.vstack([q_ref, dq_ref])
        
        # Solve MPC
        tau, _ = self.solve(x_current, x_reference)
        
        return tau


def load_trained_model(model_type, config):
    """Load trained model from checkpoint"""
    from models.baseline_dnn import create_baseline_model
    from models.transformer_predictor import create_transformer_model
    
    if model_type == 'baseline':
        model = create_baseline_model(config)
    else:
        model = create_transformer_model(config)
    
    # Load checkpoint
    checkpoint_path = Path(f'models/trained/{model_type}/best.pth')
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No trained model found at {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded trained {model_type} model from {checkpoint_path}")
    
    return model


if __name__ == "__main__":
    # Test MPC
    with open("configs/config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Create dummy model for testing
    from models.baseline_dnn import create_baseline_model
    model = create_baseline_model(config)
    
    # Create MPC
    mpc = LearnedDynamicsMPC(model, config, model_type='baseline')
    
    # Test step
    q_current = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    dq_current = np.zeros(7)
    q_ref = np.array([0.5, 0.3, 0.2, 0.1, 0.0, -0.1, -0.2])
    
    tau = mpc.step(q_current, dq_current, q_ref)
    print(f"Optimal torque: {tau}")
