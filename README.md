# Transformer-based Model Predictive Control for KUKA LBR4

Implementation of **temporal-aware data-driven MPC** using Transformer networks for precise robotic trajectory tracking, building on the work by El-Hussieny et al. (2024).

## 📋 Overview

This project implements and compares two neural architectures for predictive modeling in MPC:
- **Baseline Feed-Forward DNN** (El-Hussieny et al., 2024)
- **Transformer with Multi-Head Self-Attention** (Proposed)

The Transformer architecture captures temporal dependencies in robot motion, leading to improved prediction accuracy and control performance.

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Create conda environment
conda env create -f environment.yml
conda activate transformer-mpc-kuka

# Verify installation
python -c "import torch; import casadi; print('✓ Installation successful')"
```

### 2. Generate Synthetic Data

```bash
python src/data_generator.py
```

This generates ~20,000 samples with:
- Pick-and-place trajectories
- Circular motions
- Figure-8 patterns
- Sinusoidal movements
- Realistic sensor noise

### 3. Train Models

Train the baseline DNN:
```bash
python src/train.py --model_type baseline
```

Train the Transformer:
```bash
python src/train.py --model_type transformer
```

Monitor training with TensorBoard:
```bash
tensorboard --logdir logs
```

### 4. Evaluate Performance

```bash
python src/evaluate.py
```

This will:
- Run both controllers on test trajectories
- Generate comparison plots
- Compute performance metrics
- Save results to `figures/`

## 📁 Project Structure

```
transformer-mpc-kuka/
├── configs/
│   └── config.yaml              # Hyperparameters & settings
├── data/
│   └── synthetic_dataset.npz    # Generated training data
├── models/
│   ├── baseline_dnn.py          # Feed-forward DNN
│   ├── transformer_predictor.py # Transformer architecture
│   └── trained/                 # Saved checkpoints
│       ├── baseline/
│       └── transformer/
├── src/
│   ├── data_generator.py        # PyBullet-based data generation
│   ├── train.py                 # Training script
│   ├── mpc_controller.py        # CasADi MPC implementation
│   └── evaluate.py              # Evaluation & visualization
├── kuka_iiwa/                   # URDF files
├── figures/                     # Generated plots
├── logs/                        # TensorBoard logs
├── environment.yml              # Conda environment
└── README.md
```

## 🏗️ Architecture Details

### Transformer Model

```
Input Embedding (21 → 128)
    ↓
Positional Encoding
    ↓
Transformer Encoder (3 layers)
  • Multi-Head Attention (4 heads)
  • Feed-Forward Network (256 dim)
  • Layer Normalization
  • Dropout (0.1)
    ↓
Output Projection (128 → 14)
```

**Key Features:**
- History length: 10 timesteps
- Self-attention captures temporal patterns
- Residual connections for training stability
- ~165K parameters

### Baseline DNN

```
Input (21) → [128] → ReLU → [32] → ReLU → Output (14)
```

**Key Features:**
- Memoryless (single timestep)
- Simple feed-forward architecture
- ~4K parameters

## 🎯 MPC Formulation

The MPC optimization problem:

```
minimize:  Σ(||q - q_ref||²_W1 + ||Δτ||²_W2)

subject to:
  • x(k+1) = f_learned(x(k), τ(k))   [Neural network dynamics]
  • q_min ≤ q(k) ≤ q_max              [Joint limits]
  • τ_min ≤ τ(k) ≤ τ_max              [Torque limits]
  • dq_min ≤ dq(k) ≤ dq_max           [Velocity limits]
```

**Solver:** IPOPT via CasADi  
**Prediction Horizon:** N = 12  
**Sampling Time:** 50ms

## 📊 Expected Results

Based on the proposal, the Transformer model should achieve:

- **30-50% reduction** in prediction MSE
- **Improved tracking** on complex trajectories
- **Better generalization** to unseen patterns
- **Real-time capable** (< 50ms per control step)

## 🔧 Configuration

Edit `configs/config.yaml` to modify:

**Model Architecture:**
```yaml
transformer:
  history_length: 10
  d_model: 128
  num_heads: 4
  num_encoder_layers: 3
```

**Training:**
```yaml
training:
  batch_size: 64
  num_epochs: 100
  learning_rate: 0.001
```

**MPC:**
```yaml
mpc:
  prediction_horizon: 12
  weights:
    state: 100.0
    control: 0.01
```

## 📈 Monitoring Training

TensorBoard provides real-time monitoring:

```bash
tensorboard --logdir logs
```

View at: `http://localhost:6006`

Metrics tracked:
- Training/validation loss
- Learning rate schedule
- Gradient norms
- Prediction accuracy

## 🧪 Testing Different Scenarios

The evaluation script tests multiple scenarios:

1. **Point Stabilization**: Step changes in target positions
2. **Circular Trajectory**: Continuous circular motion in joint space
3. **Figure-8**: Complex curved path with direction changes
4. **Sinusoidal**: Multi-frequency oscillatory motion

## 🐛 Troubleshooting

### URDF Not Found
```bash
# Ensure URDF is in correct location:
ls kuka_iiwa/urdf/iiwa7.urdf
```

### IPOPT Solver Fails
```bash
# Reduce prediction horizon in config.yaml:
mpc:
  prediction_horizon: 8  # Try smaller value
```

### Out of Memory During Training
```bash
# Reduce batch size:
training:
  batch_size: 32  # Or 16
```

### Slow Training on CPU
```bash
# Check CUDA availability:
python -c "import torch; print(torch.cuda.is_available())"
```

## 📝 Key Implementation Notes

1. **Temporal Context**: Transformer uses sliding window of past 10 timesteps
2. **Positional Encoding**: Sinusoidal encoding preserves temporal order
3. **Noise Injection**: Gaussian noise added to simulate real sensors
4. **Constraint Handling**: MPC enforces physical limits via CasADi
5. **Real-time Feasibility**: Lightweight architecture for fast inference

## 🔬 Research Extensions

Potential improvements:
- [ ] LSTM comparison (mentioned in proposal)
- [ ] Attention visualization
- [ ] Longer history windows
- [ ] Hardware deployment
- [ ] Disturbance rejection tests
- [ ] Multi-robot coordination

## 📚 References

1. El-Hussieny et al. (2024). "Advancing Robotic Control: Data-Driven Model Predictive Control for a 7-DOF Robotic Manipulator." *IEEE Access*.

2. Vaswani et al. (2017). "Attention Is All You Need." *NeurIPS*.

3. Andersson et al. (2019). "CasADi: A Software Framework for Nonlinear Optimization and Optimal Control." *Mathematical Programming Computation*.

## 👥 Authors

Anushtup Nandy  
Department of Mechanical Engineering  
Columbia University

