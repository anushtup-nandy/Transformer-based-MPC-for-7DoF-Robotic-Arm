from matplotlib.pyplot import axis
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import yaml
from pathlib import Path
import sys
from tqdm import tqdm
import argparse

sys.path.append(str(Path(__file__).parent.parent))
from models.baseline_dnn import create_baseline_model
from models.transformer_predictor import create_transformer_model


class RobotDataset(Dataset):
    """Dataset for robot dynamics"""
    def __init__(self, data_path, split='train', history_length=1):
        data = np.load(data_path)
        
        self.positions = data[f'{split}_positions']
        self.velocities = data[f'{split}_velocities']
        self.torques = data[f'{split}_torques']
        self.next_positions = data[f'{split}_next_positions']
        self.next_velocities = data[f'{split}_next_velocities']
        
        self.history_length = history_length
        
        if history_length > 1:
            self._create_sequences()
    
    def _create_sequences(self):
        """Create sequences for transformer training"""
        n_samples = len(self.positions)
        n_sequences = n_samples - self.history_length + 1
        
        pos_seq = np.zeros((n_sequences, self.history_length, 7))
        vel_seq = np.zeros((n_sequences, self.history_length, 7))
        tau_seq = np.zeros((n_sequences, self.history_length, 7))
        next_pos = np.zeros((n_sequences, 7))
        next_vel = np.zeros((n_sequences, 7))
        
        for i in range(n_sequences):
            pos_seq[i] = self.positions[i:i+self.history_length]
            vel_seq[i] = self.velocities[i:i+self.history_length]
            tau_seq[i] = self.torques[i:i+self.history_length]
            next_pos[i] = self.next_positions[i+self.history_length-1]
            next_vel[i] = self.next_velocities[i+self.history_length-1]
        
        self.positions = pos_seq
        self.velocities = vel_seq
        self.torques = tau_seq
        self.next_positions = next_pos
        self.next_velocities = next_vel
    
    def __len__(self):
        return len(self.positions)
    
    def __getitem__(self, idx):
        return {
            'positions': torch.FloatTensor(self.positions[idx]),
            'velocities': torch.FloatTensor(self.velocities[idx]),
            'torques': torch.FloatTensor(self.torques[idx]),
            'next_positions': torch.FloatTensor(self.next_positions[idx]),
            'next_velocities': torch.FloatTensor(self.next_velocities[idx])
        }


class WeightedMSELoss(nn.Module):
    """MSE loss with separate weights for position and velocity"""
    
    def __init__(self, pos_weight=1.0, vel_weight=1.0):
        super().__init__()
        self.pos_weight = pos_weight
        self.vel_weight = vel_weight
    
    def forward(self, pred_pos, target_pos, pred_vel, target_vel):
        loss_pos = torch.mean((pred_pos - target_pos) ** 2)
        loss_vel = torch.mean((pred_vel - target_vel) ** 2)
        
        # Weight velocity loss less since it has much higher magnitude
        return self.pos_weight * loss_pos + self.vel_weight * loss_vel


class Trainer:
    """Trainer for robot dynamics models"""
    
    def __init__(self, model, config, model_type='baseline'):
        self.model = model
        self.config = config
        self.model_type = model_type
        
        # Setup device
        self.device = torch.device('mps' if torch.mps.is_available() else 'cpu')
        self.model.to(self.device)
        print(f"Training on device: {self.device}")
        
        # Setup optimizer
        train_config = config['training']
        if model_type == 'transformer':
            lr = train_config.get('transformer_lr', 0.0003)  # Lower LR
            weight_decay = train_config.get('transformer_wd', 1e-4)  # More regularization
        else:
            lr = train_config.get('baseline_lr', 0.001)
            weight_decay = train_config['weight_decay']
        
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Setup learning rate scheduler
        self.setup_scheduler(train_config)
        
        self.criterion = WeightedMSELoss(pos_weight=1.0, vel_weight=0.1)
        
        print(f"Using weighted loss: pos_weight=1.0, vel_weight=0.1")
        
        # Tensorboard
        if config['logging']['tensorboard']:
            log_dir = Path(config['logging']['log_dir']) / model_type
            self.writer = SummaryWriter(log_dir)
        else:
            self.writer = None
        
        # Training state
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Create checkpoint directory
        self.checkpoint_dir = Path('models/trained') / model_type
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def setup_scheduler(self, train_config):
        """Setup learning rate scheduler with warmup for transformer"""
        sched_config = train_config['scheduler']
        
        if self.model_type == 'transformer':
            # ✅ Warmup + Cosine schedule
            from torch.optim.lr_scheduler import LambdaLR
            
            warmup_epochs = sched_config.get('warmup_epochs', 5)
            total_epochs = train_config['num_epochs']
            
            def lr_lambda(epoch):
                if epoch < warmup_epochs:
                    # Linear warmup
                    return (epoch + 1) / warmup_epochs
                else:
                    # Cosine decay
                    progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
                    return 0.5 * (1 + np.cos(np.pi * progress))
            
            self.scheduler = LambdaLR(self.optimizer, lr_lambda)
        else:
            # Baseline uses simple cosine
            if sched_config['type'] == 'cosine':
                self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=train_config['num_epochs'],
                    eta_min=sched_config['min_lr']
                )   

    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_pos_loss = 0
        total_vel_loss = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {self.epoch+1}')
        for batch in pbar:
            # Move to device
            positions = batch['positions'].to(self.device)
            velocities = batch['velocities'].to(self.device)
            torques = batch['torques'].to(self.device)
            next_positions = batch['next_positions'].to(self.device)
            next_velocities = batch['next_velocities'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Handle baseline vs transformer
            if self.model_type == 'baseline':
                if positions.dim() == 3:
                    positions = positions[:, -1, :]
                    velocities = velocities[:, -1, :]
                    torques = torques[:, -1, :]
                
                predicted_state = self.model(positions, velocities, torques)
            else:
                predicted_state = self.model(positions, velocities, torques)
            
            # Split predictions
            pred_positions = predicted_state[:, :7]
            pred_velocities = predicted_state[:, 7:]
            
            # Compute weighted loss
            loss = self.criterion(pred_positions, next_positions, 
                                pred_velocities, next_velocities)
            
            # Track separate losses for monitoring
            with torch.no_grad():
                pos_loss = torch.mean((pred_positions - next_positions) ** 2)
                vel_loss = torch.mean((pred_velocities - next_velocities) ** 2)
                total_pos_loss += pos_loss.item()
                total_vel_loss += vel_loss.item()
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({
                'loss': loss.item(),
                'pos': pos_loss.item(),
                'vel': vel_loss.item()
            })
        
        avg_loss = total_loss / len(train_loader)
        avg_pos_loss = total_pos_loss / len(train_loader)
        avg_vel_loss = total_vel_loss / len(train_loader)
        
        return avg_loss, avg_pos_loss, avg_vel_loss

    def validate(self, val_loader):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        total_pos_loss = 0
        total_vel_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                positions = batch['positions'].to(self.device)
                velocities = batch['velocities'].to(self.device)
                torques = batch['torques'].to(self.device)
                next_positions = batch['next_positions'].to(self.device)
                next_velocities = batch['next_velocities'].to(self.device)
                
                # Forward pass
                if self.model_type == 'baseline':
                    if positions.dim() == 3:
                        positions = positions[:, -1, :]
                        velocities = velocities[:, -1, :]
                        torques = torques[:, -1, :]
                    
                    predicted_state = self.model(positions, velocities, torques)
                else:
                    predicted_state = self.model(positions, velocities, torques)
                
                pred_positions = predicted_state[:, :7]
                pred_velocities = predicted_state[:, 7:]
                
                loss = self.criterion(pred_positions, next_positions,
                                    pred_velocities, next_velocities)
                
                pos_loss = torch.mean((pred_positions - next_positions) ** 2)
                vel_loss = torch.mean((pred_velocities - next_velocities) ** 2)
                
                total_loss += loss.item()
                total_pos_loss += pos_loss.item()
                total_vel_loss += vel_loss.item()
        
        avg_loss = total_loss / len(val_loader)
        avg_pos_loss = total_pos_loss / len(val_loader)
        avg_vel_loss = total_vel_loss / len(val_loader)
        
        return avg_loss, avg_pos_loss, avg_vel_loss
    
    def save_checkpoint(self, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        # Save latest
        path = self.checkpoint_dir / 'latest.pth'
        torch.save(checkpoint, path)
        
        # Save best
        if is_best:
            path = self.checkpoint_dir / 'best.pth'
            torch.save(checkpoint, path)
            print(f"  ✓ Saved best model with val_loss={self.best_val_loss:.6f}")
    
    def train(self, train_loader, val_loader, num_epochs):
        """Full training loop"""
        train_config = self.config['training']
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Train
            train_loss, train_pos, train_vel = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_pos, val_vel = self.validate(val_loader)
            
            # Log
            print(f"Epoch {epoch+1}/{num_epochs}:")
            print(f"  train_loss={train_loss:.6f} (pos={train_pos:.6f}, vel={train_vel:.6f})")
            print(f"  val_loss={val_loss:.6f} (pos={val_pos:.6f}, vel={val_vel:.6f})")
            
            if self.writer:
                self.writer.add_scalar('Loss/train', train_loss, epoch)
                self.writer.add_scalar('Loss/val', val_loss, epoch)
                self.writer.add_scalar('Loss/train_pos', train_pos, epoch)
                self.writer.add_scalar('Loss/train_vel', train_vel, epoch)
                self.writer.add_scalar('Loss/val_pos', val_pos, epoch)
                self.writer.add_scalar('Loss/val_vel', val_vel, epoch)
                self.writer.add_scalar('LR', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Learning rate scheduler
            if self.scheduler:
                self.scheduler.step()
            
            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            if train_config['save_best'] and is_best:
                self.save_checkpoint(is_best=True)
            
            if (epoch + 1) % train_config['save_every'] == 0:
                self.save_checkpoint()
            
            # Early stopping
            early_stop_config = train_config['early_stopping']
            if self.patience_counter >= early_stop_config['patience']:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
        
        if self.writer:
            self.writer.close()
        
        print(f"\nTraining complete! Best val_loss: {self.best_val_loss:.6f}")


def main(args):
    """Main training function"""
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    # Determine history length
    if args.model_type == 'baseline':
        history_length = 1
    else:
        history_length = config['transformer']['history_length']
    
    # Create datasets
    print(f"\nLoading data from {args.data_path}")
    train_dataset = RobotDataset(args.data_path, 'train', history_length)
    val_dataset = RobotDataset(args.data_path, 'val', history_length)
   
    # Compute normalization statistics
    if history_length == 1:
        pos_mean = train_dataset.positions.mean(axis=0)
        pos_std = train_dataset.positions.std(axis=0) + 1e-6
        vel_mean = train_dataset.velocities.mean(axis=0)
        vel_std = train_dataset.velocities.std(axis=0) + 1e-6
        tau_mean = train_dataset.torques.mean(axis=0)
        tau_std = train_dataset.torques.std(axis=0) + 1e-6
    else:
        pos_mean = train_dataset.positions.reshape(-1, 7).mean(axis=0)
        pos_std = train_dataset.positions.reshape(-1, 7).std(axis=0) + 1e-6
        vel_mean = train_dataset.velocities.reshape(-1, 7).mean(axis=0)
        vel_std = train_dataset.velocities.reshape(-1, 7).std(axis=0) + 1e-6
        tau_mean = train_dataset.torques.reshape(-1, 7).mean(axis=0)
        tau_std = train_dataset.torques.reshape(-1, 7).std(axis=0) + 1e-6

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"\nNormalization stats:")
    print(f"  vel_std: {vel_std}")
    
    # Create dataloaders
    if args.model_type == 'transformer':
        batch_size = config['training'].get('transformer_batch_size', 128)  # Bigger batches
    else:
        batch_size = config['training']['batch_size']

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=0
    )
    
    # Create model
    print(f"\nCreating {args.model_type} model...")
    if args.model_type == 'baseline':
        model = create_baseline_model(config)
    else:
        model = create_transformer_model(config)

    model.pos_mean.copy_(torch.FloatTensor(pos_mean))
    model.pos_std.copy_(torch.FloatTensor(pos_std))
    model.vel_mean.copy_(torch.FloatTensor(vel_mean))
    model.vel_std.copy_(torch.FloatTensor(vel_std))
    model.tau_mean.copy_(torch.FloatTensor(tau_mean))
    model.tau_std.copy_(torch.FloatTensor(tau_std))
    
    # Create trainer
    trainer = Trainer(model, config, args.model_type)
    
    # Train
    print("\nStarting training...")
    trainer.train(train_loader, val_loader, config['training']['num_epochs'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train robot dynamics model')
    parser.add_argument('--model_type', type=str, default='transformer',
                        choices=['baseline', 'transformer'],
                        help='Model type to train')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    parser.add_argument('--data_path', type=str, default='data/synthetic_dataset.npz',
                        help='Path to dataset')
    
    args = parser.parse_args()
    main(args)
