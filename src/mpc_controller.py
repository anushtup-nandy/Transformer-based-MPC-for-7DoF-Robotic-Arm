"""
Model Predictive Controller - FIXED + LSTM-SUPPORTED VERSION
Uses soft constraints and proper optimization formulation
"""

import casadi as ca  # kept for parity with original imports (not used directly here)
import numpy as np
import torch
import yaml
from pathlib import Path
import sys
from collections import deque

sys.path.append(str(Path(__file__).parent.parent))


class LearnedDynamicsMPC:
    """MPC controller using learned neural network dynamics"""

    def __init__(self, model, config, model_type='transformer'):
        self.model = model
        self.model.eval()
        self.device = torch.device('cpu')  # CPU
        self.model.to(self.device)

        self.config = config
        self.model_type = model_type  # 'baseline' | 'transformer' | 'lstm'

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

        # History configuration:
        # sequence models (transformer, lstm) consume T-step histories; baseline uses last step only.
        if self.model_type in ('transformer', 'lstm'):
            self.history_length = config['transformer']['history_length']
        else:
            self.history_length = 1

        # Deques for history (shared by transformer and lstm)
        self.position_history = deque(maxlen=self.history_length)
        self.velocity_history = deque(maxlen=self.history_length)
        self.torque_history   = deque(maxlen=self.history_length)

        # Build MPC problem boxes
        self.build_mpc()

        print(f"✓ MPC initialized with horizon N={self.N}")
        print(f"✓ Using {model_type} model for dynamics prediction (history_length={self.history_length})")

    # -------------------- History helpers --------------------
    def reset_history(self):
        """Reset history buffer (sequence models)"""
        self.position_history.clear()
        self.velocity_history.clear()
        self.torque_history.clear()

    def update_history(self, q, dq, tau):
        """Append to history (sequence models)"""
        self.position_history.append(q.copy())
        self.velocity_history.append(dq.copy())
        self.torque_history.append(tau.copy())

    def _ensure_history_initialized(self, q, dq, tau):
        """
        Make sure we have 'history_length' samples by padding the
        current state when starting up.
        """
        deficit = self.history_length - len(self.position_history)
        for _ in range(max(0, deficit)):
            self.position_history.appendleft(q.copy())
            self.velocity_history.appendleft(np.zeros_like(dq))
            self.torque_history.appendleft(np.zeros_like(tau))

    # -------------------- Learned one-step rollout --------------------
    def predict_next_state_numpy(self, q, dq, tau):
        """
        Predict next state using the learned dynamics model.
        - baseline: uses only last step (B=1)
        - transformer: uses (B=1, T, 7) for q, dq, tau
        - lstm: concatenates to (B=1, T, 21) and calls model(x_seq)
        """
        with torch.no_grad():
            if self.model_type == 'baseline':
                q_tensor  = torch.FloatTensor(q).unsqueeze(0).to(self.device)   # (1,7)
                dq_tensor = torch.FloatTensor(dq).unsqueeze(0).to(self.device)  # (1,7)
                tau_tensor= torch.FloatTensor(tau).unsqueeze(0).to(self.device) # (1,7)
                next_state = self.model(q_tensor, dq_tensor, tau_tensor)        # (1,14)
                next_state = next_state[0].cpu().numpy()

            elif self.model_type == 'transformer':
                # ensure history is populated
                self._ensure_history_initialized(q, dq, tau)
                # append this sample as the most recent
                self.update_history(q, dq, tau)

                q_seq   = torch.from_numpy(np.stack(self.position_history, axis=0)).float().unsqueeze(0).to(self.device)   # (1,T,7)
                dq_seq  = torch.from_numpy(np.stack(self.velocity_history, axis=0)).float().unsqueeze(0).to(self.device)   # (1,T,7)
                tau_seq = torch.from_numpy(np.stack(self.torque_history,   axis=0)).float().unsqueeze(0).to(self.device)   # (1,T,7)
                next_state = self.model(q_seq, dq_seq, tau_seq)  # (1,14)
                next_state = next_state[0].cpu().numpy()

            elif self.model_type == 'lstm':
                # ensure history is populated
                self._ensure_history_initialized(q, dq, tau)
                # append this sample as the most recent
                self.update_history(q, dq, tau)

                q_seq   = torch.from_numpy(np.stack(self.position_history, axis=0)).float().unsqueeze(0).to(self.device)   # (1,T,7)
                dq_seq  = torch.from_numpy(np.stack(self.velocity_history, axis=0)).float().unsqueeze(0).to(self.device)   # (1,T,7)
                tau_seq = torch.from_numpy(np.stack(self.torque_history,   axis=0)).float().unsqueeze(0).to(self.device)   # (1,T,7)
                x_seq = torch.cat([q_seq, dq_seq, tau_seq], dim=-1)  # (1,T,21)
                next_state = self.model(x_seq)  # (1,14)
                next_state = next_state[0].cpu().numpy()

            else:
                raise ValueError(f"Unknown model_type: {self.model_type}")

        q_next  = next_state[:7]
        dq_next = next_state[7:]
        return q_next, dq_next

    # -------------------- MPC boxes --------------------
    def build_mpc(self):
        """Build MPC problem dimensions and box constraints"""
        # Total variables: (N+1)*14 states + N*7 controls
        n_states = (self.N + 1) * 14
        n_controls = self.N * 7
        n_vars = n_states + n_controls

        # Use RELAXED bounds
        lbx = []
        ubx = []

        # State bounds
        for _ in range(self.N + 1):
            lbx.extend(self.q_min * 0.99)
            lbx.extend(self.dq_min * 0.99)
            ubx.extend(self.q_max * 0.99)
            ubx.extend(self.dq_max * 0.99)

        # Control bounds
        for _ in range(self.N):
            lbx.extend(self.tau_min * 0.95)
            ubx.extend(self.tau_max * 0.95)

        self.lbx = np.array(lbx)
        self.ubx = np.array(ubx)
        self.n_vars = n_vars
        self.n_states = n_states
        self.n_controls = n_controls

        print(f"✓ MPC problem built: {n_vars} decision variables")

    # -------------------- Objective (soft dynamics) --------------------
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

        # Tracking + smoothness cost
        cost = 0.0
        for k in range(self.N):
            q_k  = states[k, :7]
            dq_k = states[k, 7:]
            tau_k = controls[k]

            q_ref_k  = x_reference[:7, k]
            dq_ref_k = x_reference[7:, k]
            e_q  = q_k - q_ref_k
            e_dq = dq_k - dq_ref_k

            cost += self.W1 * (np.dot(e_q, e_q) + np.dot(e_dq, e_dq))

            if k > 0:
                delta_tau = tau_k - controls[k-1]
                cost += self.W2 * np.dot(delta_tau, delta_tau)

        # Terminal cost
        e_qf  = states[self.N, :7] - x_reference[:7, self.N]
        e_dqf = states[self.N, 7:] - x_reference[7:, self.N]
        cost += 2.0 * self.W1 * (np.dot(e_qf, e_qf) + np.dot(e_dqf, e_dqf))

        # Initial condition penalty (hard via big penalty)
        init_error = states[0] - x_current
        cost += 10000.0 * np.dot(init_error, init_error)

        # Prepare rollout histories for sequence models
        is_seq = (self.model_type in ('transformer', 'lstm'))
        if is_seq:
            rollout_q   = deque(self.position_history, maxlen=self.history_length)
            rollout_dq  = deque(self.velocity_history, maxlen=self.history_length)
            rollout_tau = deque(self.torque_history,   maxlen=self.history_length)
            if len(rollout_q) < self.history_length:
                raise RuntimeError(
                    f"History not initialized! length={len(rollout_q)}, expected={self.history_length}"
                )

        # Soft dynamics constraints along horizon
        for k in range(self.N):
            q_k  = states[k, :7]
            dq_k = states[k, 7:]
            tau_k = controls[k]

            if is_seq:
                # build tensors (1, T, 7) and call appropriate model
                q_seq   = torch.from_numpy(np.stack(rollout_q,  axis=0)).float().unsqueeze(0).to(self.device)
                dq_seq  = torch.from_numpy(np.stack(rollout_dq, axis=0)).float().unsqueeze(0).to(self.device)
                tau_seq = torch.from_numpy(np.stack(rollout_tau,axis=0)).float().unsqueeze(0).to(self.device)

                with torch.no_grad():
                    if self.model_type == 'transformer':
                        pred = self.model(q_seq, dq_seq, tau_seq)  # (1,14)
                    else:  # lstm
                        x_seq = torch.cat([q_seq, dq_seq, tau_seq], dim=-1)  # (1,T,21)
                        pred = self.model(x_seq)  # (1,14)

                pred_np = pred[0].cpu().numpy()
                q_next_pred  = pred_np[:7]
                dq_next_pred = pred_np[7:]

                # roll histories forward with current control guess
                rollout_q.append(q_next_pred)
                rollout_dq.append(dq_next_pred)
                rollout_tau.append(tau_k)

            else:
                # baseline one-step prediction
                q_next_pred, dq_next_pred = self.predict_next_state_numpy(q_k, dq_k, tau_k)

            next_state_pred = np.concatenate([q_next_pred, dq_next_pred])
            dyn_err = states[k+1] - next_state_pred
            cost += self.W_dynamics * np.dot(dyn_err, dyn_err)

        # Soft bound penalties (quadratic near bounds)
        for k in range(self.N + 1):
            q_k  = states[k, :7]
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

    # -------------------- Solve & Step --------------------
    def solve(self, x_current, x_reference):
        """
        Solve MPC using scipy optimization (box constraints)
        """
        from scipy.optimize import minimize

        X0 = np.zeros(self.n_vars)

        # Initialize state trajectory by linear interpolation
        for k in range(self.N + 1):
            alpha = k / self.N
            X0[k*14:(k+1)*14] = (1 - alpha) * x_current + alpha * x_reference[:, min(k, x_reference.shape[1] - 1)]

        # Small random control init
        X0[self.n_states:] = np.random.randn(self.n_controls) * 0.01

        # Clip to box bounds
        X0 = np.clip(X0, self.lbx, self.ubx)

        result = minimize(
            lambda X: self.objective_with_soft_constraints(X, x_current, x_reference),
            X0,
            method='L-BFGS-B',
            bounds=list(zip(self.lbx, self.ubx)),
            options={
                'maxiter': 100,
                'ftol': 1e-6,
                'gtol': 1e-5,
                'disp': False
            }
        )

        if result.success or result.fun < 1e10:
            X_opt = result.x
            states_opt   = X_opt[:self.n_states].reshape(self.N + 1, 14)
            controls_opt = X_opt[self.n_states:].reshape(self.N, 7)
            return controls_opt[0], states_opt.T
        else:
            print(f"⚠ Optimization failed: {result.message} (fun={result.fun:.2e})")
            return np.zeros(self.dof), None

    def step(self, q_current, dq_current, q_ref, dq_ref=None):
        """Single MPC control step"""
        # Prepare current full state
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

        # Ensure/initialize history for sequence models
        if self.model_type in ('transformer', 'lstm'):
            self._ensure_history_initialized(q_current, dq_current, np.zeros(self.dof))
            # Optional: prime history with current state if it's new
            if len(self.position_history) == 0 or not np.allclose(self.position_history[-1], q_current):
                self.update_history(q_current, dq_current, np.zeros(self.dof))

        # Solve MPC
        tau, _ = self.solve(x_current, x_reference)

        # Update history with the control we just computed (sequence models)
        if self.model_type in ('transformer', 'lstm'):
            self.update_history(q_current, dq_current, tau)

        return tau


# -------------------- Utility: load trained model --------------------
def load_trained_model(model_type, config):
    """
    Load a trained model from models/trained/<model_type>/best.pth
    Supports baseline / transformer / lstm
    """
    from models.baseline_dnn import create_baseline_model
    from models.transformer_predictor import create_transformer_model
    try:
        from models.lstm_predictor import create_lstm_model
    except Exception:
        create_lstm_model = None

    if model_type == 'baseline':
        model = create_baseline_model(config)
    elif model_type == 'transformer':
        model = create_transformer_model(config)
    elif model_type == 'lstm':
        if create_lstm_model is None:
            raise ImportError("models.lstm_predictor.create_lstm_model not found")
        # Default LSTM hyperparams (match train/eval script defaults)
        model = create_lstm_model(config, hidden_dim=128, num_layers=2, dropout=0.1)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    checkpoint_path = Path(f'models/trained/{model_type}/best.pth')
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No trained model found at {checkpoint_path}")

    state = torch.load(checkpoint_path, map_location='cpu')
    if isinstance(state, dict) and 'model_state_dict' in state:
        model.load_state_dict(state['model_state_dict'])
    else:
        model.load_state_dict(state)

    model.eval()
    print(f"✓ Loaded trained {model_type} model from {checkpoint_path}")

    return model
