"""
Training script for both Baseline DNN and Transformer models
"""

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
        """
        Args:
            data_path: Path to .npz file
            split: 'train', 'val', or 'test'
            history_length: Number of past timesteps (1 for DNN, >1 for Transformer)
        """
        data = np.load(data_path)
        
        self.positions = data[f'{split}_positions']
        self.velocities = data[f'{split}_velocities']
        self.torques = data[f'{split}_torques']
        self.next_positions = data[f'{split}_next_positions']
        self.next_velocities = data[f'{split}_next_velocities']
        
        self.history_length = history_length
        
        # For transformer, we need to create sequences
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
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=train_config['learning_rate'],
            weight_decay=train_config['weight_decay']
        )
        
        # Setup learning rate scheduler
        self.setup_scheduler(train_config)
        
        # Loss function
        self.criterion = nn.MSELoss()
        
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
        """Setup learning rate scheduler"""
        sched_config = train_config['scheduler']
        
        if sched_config['type'] == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=train_config['num_epochs'],
                eta_min=sched_config['min_lr']
            )
        elif sched_config['type'] == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=20,
                gamma=0.5
            )
        elif sched_config['type'] == 'exponential':
            self.scheduler = optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=0.95
            )
        else:
            self.scheduler = None
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {self.epoch+1}')
        for batch in pbar:
            # Move to device
            positions = batch['positions'].to(self.device)
            velocities = batch['velocities'].to(self.device)
            torques = batch['torques'].to(self.device)
            next_positions = batch['next_positions'].to(self.device)
            next_velocities = batch['next_velocities'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            if self.model_type == 'baseline':
                # Baseline DNN: single timestep input
                if positions.dim() == 3:  # If accidentally has sequence dimension
                    positions = positions[:, -1, :]
                    velocities = velocities[:, -1, :]
                    torques = torques[:, -1, :]
                
                predicted_state = self.model(positions, velocities, torques)
            else:
                # Transformer: sequence input
                predicted_state = self.model(positions, velocities, torques)
            
            # Split predictions
            pred_positions = predicted_state[:, :7]
            pred_velocities = predicted_state[:, 7:]
            
            # Compute loss
            loss_pos = self.criterion(pred_positions, next_positions)
            loss_vel = self.criterion(pred_velocities, next_velocities)
            loss = loss_pos + loss_vel
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        
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
                
                loss_pos = self.criterion(pred_positions, next_positions)
                loss_vel = self.criterion(pred_velocities, next_velocities)
                loss = loss_pos + loss_vel
                
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
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
            print(f"  Saved best model with val_loss={self.best_val_loss:.6f}")
    
    def train(self, train_loader, val_loader, num_epochs):
        """Full training loop"""
        train_config = self.config['training']
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss = self.validate(val_loader)
            
            # Log
            print(f"Epoch {epoch+1}/{num_epochs}: "
                  f"train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")
            
            if self.writer:
                self.writer.add_scalar('Loss/train', train_loss, epoch)
                self.writer.add_scalar('Loss/val', val_loss, epoch)
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
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set random seed
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
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
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
