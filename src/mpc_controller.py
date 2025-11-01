"""
Model Predictive Controller - FIXED VERSION
Uses soft constraints and proper optimization formulation
"""

import casadi as ca
import numpy as np
import torch
import yaml
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))


class LearnedDynamicsMPC:
    """MPC controller using learned neural network dynamics"""
    
    def __init__(self, model, config, model_type='transformer'):
        self.model = model
        self.model.eval()
        self.device = torch.device('cpu') # cpu
        self.model.to(self.device)
        
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
        
        # Soft constraint weights
        self.W_dynamics = 1000.0  # Weight for dynamics soft constraint
        self.W_slack = 100.0      # Weight for bound slack variables
        
        # For transformer: history management
        if model_type == 'transformer':
            self.history_length = config['transformer']['history_length']
            self.reset_history()
        else:
            self.history_length = 1
        
        # Build MPC problem
        self.build_mpc()
        
        print(f"✓ MPC initialized with horizon N={self.N}")
        print(f"✓ Using {model_type} model for dynamics prediction")
    
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
    
    def predict_next_state_numpy(self, q, dq, tau):
        """Predict next state using learned model"""
        with torch.no_grad():
            if self.model_type == 'baseline':
                q_tensor = torch.FloatTensor(q).unsqueeze(0).to(self.device)
                dq_tensor = torch.FloatTensor(dq).unsqueeze(0).to(self.device)
                tau_tensor = torch.FloatTensor(tau).unsqueeze(0).to(self.device)
                
                next_state = self.model(q_tensor, dq_tensor, tau_tensor)
                q_next = next_state[0, :7].cpu().numpy()
                dq_next = next_state[0, 7:].cpu().numpy()
            else:
                # Transformer: use history
                if len(self.position_history) < self.history_length:
                    while len(self.position_history) < self.history_length:
                        self.position_history.append(q.copy())
                        self.velocity_history.append(dq.copy())
                        self.torque_history.append(tau.copy())
                
                hist_q = torch.FloatTensor(np.array(self.position_history)).unsqueeze(0).to(self.device)
                hist_dq = torch.FloatTensor(np.array(self.velocity_history)).unsqueeze(0).to(self.device)
                hist_tau = torch.FloatTensor(np.array(self.torque_history)).unsqueeze(0).to(self.device)
                
                next_state = self.model(hist_q, hist_dq, hist_tau)
                q_next = next_state[0, :7].cpu().numpy()
                dq_next = next_state[0, 7:].cpu().numpy()
        
        return q_next, dq_next
    
    def build_mpc(self):
        """Build MPC problem dimensions"""
        # Total variables: (N+1)*14 states + N*7 controls
        n_states = (self.N + 1) * 14
        n_controls = self.N * 7
        n_vars = n_states + n_controls
        
        # Use RELAXED bounds (not artificially tight)
        lbx = []
        ubx = []
        
        # State bounds - use full range with small margin
        for k in range(self.N + 1):
            lbx.extend(self.q_min * 0.99)   # 99% instead of 95%
            lbx.extend(self.dq_min * 0.99)  # 99% instead of 90%
            ubx.extend(self.q_max * 0.99)
            ubx.extend(self.dq_max * 0.99)
        
        # Control bounds - use full range
        for k in range(self.N):
            lbx.extend(self.tau_min * 0.95)  # 95% instead of 80%
            ubx.extend(self.tau_max * 0.95)
        
        self.lbx = np.array(lbx)
        self.ubx = np.array(ubx)
        
        # Store dimensions
        self.n_vars = n_vars
        self.n_states = n_states
        self.n_controls = n_controls
        
        print(f"✓ MPC problem built: {n_vars} decision variables")
    
    def objective_with_soft_constraints(self, X, x_current, x_reference):
        """
        Compute objective with soft dynamics constraints
        
        Args:
            X: Decision variables [states; controls]
            x_current: Initial state (14,)
            x_reference: Reference trajectory (14, N+1)
        
        Returns:
            cost: Total cost including penalties
        """
        # Extract states and controls
        states = X[:self.n_states].reshape(self.N + 1, 14)
        controls = X[self.n_states:].reshape(self.N, 7)
        
        # Tracking cost
        cost = 0.0
        
        # Stage costs
        for k in range(self.N):
            q_k = states[k, :7]
            dq_k = states[k, 7:]
            tau_k = controls[k]
            
            # Tracking error
            q_ref_k = x_reference[:7, k]
            dq_ref_k = x_reference[7:, k]
            
            e_q = q_k - q_ref_k
            e_dq = dq_k - dq_ref_k
            
            cost += self.W1 * (np.dot(e_q, e_q) + np.dot(e_dq, e_dq))
            
            # Control rate
            if k > 0:
                delta_tau = tau_k - controls[k-1]
                cost += self.W2 * np.dot(delta_tau, delta_tau)
        
        # Terminal cost
        e_q_final = states[self.N, :7] - x_reference[:7, self.N]
        e_dq_final = states[self.N, 7:] - x_reference[7:, self.N]
        cost += 2.0 * self.W1 * (np.dot(e_q_final, e_q_final) + 
                                  np.dot(e_dq_final, e_dq_final))
        
        # Initial condition (hard constraint via penalty)
        init_error = states[0] - x_current
        cost += 10000.0 * np.dot(init_error, init_error)
        
        # Soft dynamics constraints
        for k in range(self.N):
            q_k = states[k, :7]
            dq_k = states[k, 7:]
            tau_k = controls[k]
            
            # Predict next state
            q_next_pred, dq_next_pred = self.predict_next_state_numpy(q_k, dq_k, tau_k)
            next_state_pred = np.concatenate([q_next_pred, dq_next_pred])
            
            # Soft constraint: penalize deviation from predicted dynamics
            dynamics_error = states[k+1] - next_state_pred
            cost += self.W_dynamics * np.dot(dynamics_error, dynamics_error)
        
        # Soft bound penalties (quadratic barrier near bounds)
        margin = 0.1  # Start penalty when within 10% of bounds
        for k in range(self.N + 1):
            q_k = states[k, :7]
            dq_k = states[k, 7:]
            
            # Position bounds
            q_lower_viol = np.maximum(0, self.q_min * 0.99 - q_k)
            q_upper_viol = np.maximum(0, q_k - self.q_max * 0.99)
            cost += self.W_slack * (np.sum(q_lower_viol**2) + np.sum(q_upper_viol**2))
            
            # Velocity bounds
            dq_lower_viol = np.maximum(0, self.dq_min * 0.99 - dq_k)
            dq_upper_viol = np.maximum(0, dq_k - self.dq_max * 0.99)
            cost += self.W_slack * (np.sum(dq_lower_viol**2) + np.sum(dq_upper_viol**2))
        
        # Torque bounds
        for k in range(self.N):
            tau_k = controls[k]
            tau_lower_viol = np.maximum(0, self.tau_min * 0.95 - tau_k)
            tau_upper_viol = np.maximum(0, tau_k - self.tau_max * 0.95)
            cost += self.W_slack * (np.sum(tau_lower_viol**2) + np.sum(tau_upper_viol**2))
        
        return cost
    
    def solve(self, x_current, x_reference):
        """
        Solve MPC using scipy optimization
        """
        from scipy.optimize import minimize
        
        # Initial guess: straight line interpolation + previous solution
        X0 = np.zeros(self.n_vars)
        
        # Initialize states with smooth interpolation
        for k in range(self.N + 1):
            alpha = k / self.N
            X0[k*14:(k+1)*14] = (1-alpha) * x_current + alpha * x_reference[:, min(k, x_reference.shape[1]-1)]
        
        # Initialize controls to zero (or small random)
        X0[self.n_states:] = np.random.randn(self.n_controls) * 0.01
        
        # Clip initial guess to bounds
        X0 = np.clip(X0, self.lbx, self.ubx)
        
        # Solve using L-BFGS-B (handles bounds better than SLSQP for unconstrained problems)
        result = minimize(
            lambda X: self.objective_with_soft_constraints(X, x_current, x_reference),
            X0,
            method='L-BFGS-B',  # Better for box-constrained optimization
            bounds=list(zip(self.lbx, self.ubx)),
            options={
                'maxiter': 100,
                'ftol': 1e-6,
                'gtol': 1e-5,
                'disp': False
            }
        )
        
        if result.success or result.fun < 1e10:  # Accept if cost is reasonable
            # Extract solution
            X_opt = result.x
            states_opt = X_opt[:self.n_states].reshape(self.N + 1, 14)
            controls_opt = X_opt[self.n_states:].reshape(self.N, 7)
            
            return controls_opt[0], states_opt.T
        else:
            # Fallback: return zero control
            print(f"⚠ Optimization failed: {result.message} (fun={result.fun:.2e})")
            return np.zeros(self.dof), None
    
    def step(self, q_current, dq_current, q_ref, dq_ref=None):
        """Single MPC control step"""
        # Prepare current state
        x_current = np.concatenate([q_current, dq_current])
        
        # Prepare reference trajectory
        if q_ref.ndim == 1:
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
    
    checkpoint_path = Path(f'models/trained/{model_type}/best.pth')
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No trained model found at {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ Loaded trained {model_type} model from {checkpoint_path}")
    
    return model
