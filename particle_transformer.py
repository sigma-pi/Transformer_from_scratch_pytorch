import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import seaborn as sns

class ParticleDataset(Dataset):
    """Dataset for particle simulation trajectories"""
    
    def __init__(self, npz_file, sequence_length=10):
        """
        Args:
            npz_file: Path to the .npz file containing simulation data
            sequence_length: Number of consecutive frames to use for training
        """
        self.data = np.load(npz_file, allow_pickle=True)
        self.sequence_length = sequence_length
        
        # Extract trajectories
        self.trajectories = []
        trajectory_idx = 0
        
        while f'simulation_trajectory_{trajectory_idx}' in self.data:
            traj_data = self.data[f'simulation_trajectory_{trajectory_idx}']
            positions = traj_data[0]  # Shape: (timesteps, particles, 2)
            materials = traj_data[1]  # Shape: (particles,)
            
            # Compute velocities using discrete differentiation
            velocities = np.zeros_like(positions)
            velocities[1:] = positions[1:] - positions[:-1]  # v_t = pos_t - pos_{t-1}
            velocities[0] = velocities[1]  # Copy first velocity
            
            # Combine positions and velocities: [pos_x, pos_y, vel_x, vel_y]
            state = np.concatenate([positions, velocities], axis=-1)  # (timesteps, particles, 4)
            
            self.trajectories.append(state)
            trajectory_idx += 1
        
        print(f"Loaded {len(self.trajectories)} trajectories")
        if len(self.trajectories) > 0:
            print(f"First trajectory shape: {self.trajectories[0].shape}")
    
    def __len__(self):
        total_samples = 0
        for traj in self.trajectories:
            total_samples += max(0, traj.shape[0] - self.sequence_length)
        return total_samples
    
    def __getitem__(self, idx):
        # Find which trajectory and timestep this index corresponds to
        current_idx = idx
        for traj in self.trajectories:
            max_start = max(0, traj.shape[0] - self.sequence_length)
            if current_idx < max_start:
                start_t = current_idx
                sequence = traj[start_t:start_t + self.sequence_length]
                
                # Return input (first seq_len-1 frames) and target (last seq_len-1 frames)
                x = torch.FloatTensor(sequence[:-1])  # (seq_len-1, particles, 4)
                y = torch.FloatTensor(sequence[1:])   # (seq_len-1, particles, 4)
                
                return x, y
            current_idx -= max_start
        
        raise IndexError("Dataset index out of range")
    
    def get_full_trajectory(self, traj_idx):
        """Get a complete trajectory for rollout evaluation"""
        if traj_idx >= len(self.trajectories):
            raise IndexError(f"Trajectory index {traj_idx} out of range")
        return torch.FloatTensor(self.trajectories[traj_idx])

def add_boundary_features(x, bounds=(0.1, 0.9)):
    """Add distance-to-boundary features"""
    positions = x[:, :, :2]  # Extract positions (batch, particles, 2)
    min_bound, max_bound = bounds
    
    # Distance to each boundary
    dist_to_min = positions - min_bound  # Distance to left/bottom
    dist_to_max = max_bound - positions  # Distance to right/top
    
    boundary_features = torch.cat([dist_to_min, dist_to_max], dim=-1)
    return torch.cat([x, boundary_features], dim=-1)  # (batch, particles, 8)

def apply_boundary_constraints(positions, velocities, bounds=(0.1, 0.9), damping=0.8):
    """Apply hard boundary constraints with reflection and damping"""
    min_bound, max_bound = bounds
    
    # Clamp positions to boundaries
    positions_clamped = torch.clamp(positions, min_bound, max_bound)
    
    # Detect boundary collisions
    hit_min = positions <= min_bound
    hit_max = positions >= max_bound
    
    # Reflect and damp velocities at boundaries
    velocities_new = velocities.clone()
    velocities_new = torch.where(hit_min, torch.abs(velocities) * damping, velocities_new)
    velocities_new = torch.where(hit_max, -torch.abs(velocities) * damping, velocities_new)
    
    return positions_clamped, velocities_new

class PhysicsInformedParticleTransformer(nn.Module):
    """Transformer-based model for particle dynamics prediction"""
    
    def __init__(self, d_model=128, n_heads=8, n_layers=3, dropout=0.1, 
                 gravity=0.0, bounds=(0.1, 0.9), dt=0.01):
        super().__init__()
        
        # Model parameters
        self.d_model = d_model
        self.bounds = bounds
        self.dt = dt
        
        # Input: [pos_x, pos_y, vel_x, vel_y, dist_to_min_x, dist_to_min_y, dist_to_max_x, dist_to_max_y]
        self.input_dim = 8
        
        # Learnable gravity (can be fine-tuned during training)
        self.gravity = nn.Parameter(torch.tensor([0.0, gravity]))
        
        # Network layers
        self.input_projection = nn.Linear(self.input_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1000, d_model))  # Max 1000 particles
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Output projection to [pos_x, pos_y, vel_x, vel_y]
        self.output_projection = nn.Linear(d_model, 4)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, num_particles, 4)
               Contains [pos_x, pos_y, vel_x, vel_y] for each particle
        
        Returns:
            Predicted next state of shape (batch_size, num_particles, 4)
        """
        batch_size, num_particles, _ = x.shape
        
        # Add boundary features
        x_augmented = add_boundary_features(x, self.bounds)  # (batch, particles, 8)
        
        # Project to model dimension
        embeddings = self.input_projection(x_augmented)  # (batch, particles, d_model)
        
        # Add positional embeddings (particle-specific, not spatial)
        embeddings = embeddings + self.pos_embedding[:num_particles].unsqueeze(0)
        embeddings = self.layer_norm(embeddings)
        
        # Apply transformer layers
        transformed = self.transformer(embeddings)  # (batch, particles, d_model)
        
        # Project to output space
        output = self.output_projection(transformed)  # (batch, particles, 4)
        
        # Split into positions and velocities
        pred_pos = output[:, :, :2]
        pred_vel = output[:, :, 2:]
        
        # Apply physics: add gravity to velocity predictions
        gravity_effect = self.gravity.unsqueeze(0).unsqueeze(0) * self.dt
        pred_vel = pred_vel + gravity_effect
        
        # Apply boundary constraints
        pred_pos, pred_vel = apply_boundary_constraints(pred_pos, pred_vel, self.bounds)
        
        # Combine and return
        return torch.cat([pred_pos, pred_vel], dim=-1)
    
    def multi_step_rollout(self, initial_state, num_steps, device='cpu'):
        """
        Perform multi-step rollout prediction
        
        Args:
            initial_state: Initial particle state (1, num_particles, 4)
            num_steps: Number of steps to predict
            device: Device to run on
        
        Returns:
            rollout: Predicted trajectory (num_steps+1, num_particles, 4)
        """
        self.eval()
        rollout = [initial_state.clone()]
        current_state = initial_state.to(device)
        
        with torch.no_grad():
            for step in range(num_steps):
                next_state = self.forward(current_state)
                rollout.append(next_state.cpu())
                current_state = next_state
        
        return torch.cat(rollout, dim=0)  # (num_steps+1, num_particles, 4)

def physics_informed_loss(pred, target, gravity_weight=0.1, boundary_weight=0.1, 
                         bounds=(0.1, 0.9), dt=0.01):
    """Combined loss function with physics constraints"""
    
    # Extract positions and velocities
    pred_pos, pred_vel = pred[:, :, :2], pred[:, :, 2:]
    true_pos, true_vel = target[:, :, :2], target[:, :, 2:]
    
    # Standard MSE losses
    pos_loss = F.mse_loss(pred_pos, true_pos)
    vel_loss = F.mse_loss(pred_vel, true_vel)
    
    # Boundary violation penalty
    min_bound, max_bound = bounds
    margin = 0.01
    
    violation_min = torch.relu(min_bound + margin - pred_pos)
    violation_max = torch.relu(pred_pos - (max_bound - margin))
    boundary_loss = (violation_min + violation_max).mean()
    
    # Combine losses
    total_loss = pos_loss + vel_loss + boundary_weight * boundary_loss
    
    return {
        'total_loss': total_loss,
        'pos_loss': pos_loss,
        'vel_loss': vel_loss,
        'boundary_loss': boundary_loss
    }

def evaluate_rollout_accuracy(model, dataset, num_rollouts=5, rollout_length=50, device='cpu'):
    """
    Evaluate model accuracy on multi-step rollouts
    
    Args:
        model: Trained model
        dataset: ParticleDataset instance
        num_rollouts: Number of trajectories to evaluate
        rollout_length: Length of rollout prediction
        device: Device to run on
    
    Returns:
        metrics: Dictionary containing evaluation metrics
    """
    model.eval()
    model.to(device)
    
    position_errors = []
    velocity_errors = []
    
    for i in range(min(num_rollouts, len(dataset.trajectories))):
        # Get ground truth trajectory
        true_trajectory = dataset.get_full_trajectory(i)
        
        if true_trajectory.shape[0] <= rollout_length:
            continue
            
        # Initial state
        initial_state = true_trajectory[0:1]  # (1, particles, 4)
        
        # Ground truth for comparison
        true_rollout = true_trajectory[:rollout_length+1]  # (rollout_length+1, particles, 4)
        
        # Model prediction
        pred_rollout = model.multi_step_rollout(initial_state, rollout_length, device)
        
        # Compute errors
        pred_positions = pred_rollout[:, :, :2]  # (steps, particles, 2)
        true_positions = true_rollout[:, :, :2]
        
        pred_velocities = pred_rollout[:, :, 2:]  # (steps, particles, 2)
        true_velocities = true_rollout[:, :, 2:]
        
        # MSE at each timestep
        pos_mse = torch.mean((pred_positions - true_positions) ** 2, dim=(1, 2))  # (steps,)
        vel_mse = torch.mean((pred_velocities - true_velocities) ** 2, dim=(1, 2))  # (steps,)
        
        position_errors.append(pos_mse)
        velocity_errors.append(vel_mse)
    
    # Average across rollouts
    avg_pos_error = torch.stack(position_errors).mean(dim=0)  # (steps,)
    avg_vel_error = torch.stack(velocity_errors).mean(dim=0)  # (steps,)
    
    return {
        'position_mse': avg_pos_error,
        'velocity_mse': avg_vel_error,
        'final_position_mse': avg_pos_error[-1].item(),
        'final_velocity_mse': avg_vel_error[-1].item()
    }

def visualize_particle_trajectory(true_trajectory, pred_trajectory, bounds=(0.1, 0.9), 
                                 save_path=None, show_every=5):
    """
    Create an animated visualization of particle trajectories
    
    Args:
        true_trajectory: Ground truth trajectory (timesteps, particles, 4)
        pred_trajectory: Predicted trajectory (timesteps, particles, 4)
        bounds: Simulation boundaries
        save_path: Path to save animation (optional)
        show_every: Show every N-th frame for performance
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Extract positions
    true_pos = true_trajectory[:, :, :2].numpy()
    pred_pos = pred_trajectory[:, :, :2].numpy()
    
    min_bound, max_bound = bounds
    
    def setup_axis(ax, title):
        ax.set_xlim(min_bound - 0.05, max_bound + 0.05)
        ax.set_ylim(min_bound - 0.05, max_bound + 0.05)
        ax.set_aspect('equal')
        ax.set_title(title)
        ax.add_patch(Rectangle((min_bound, min_bound), 
                              max_bound - min_bound, max_bound - min_bound,
                              fill=False, edgecolor='black', linewidth=2))
        return ax.scatter([], [], s=20, alpha=0.7)
    
    scat1 = setup_axis(ax1, 'Ground Truth')
    scat2 = setup_axis(ax2, 'Model Prediction')
    
    def animate(frame):
        if frame >= len(true_pos):
            return scat1, scat2
            
        # Update scatter plots
        scat1.set_offsets(true_pos[frame])
        scat2.set_offsets(pred_pos[frame])
        
        # Color particles by velocity magnitude for visual appeal
        true_vel = true_trajectory[frame, :, 2:].numpy()
        pred_vel = pred_trajectory[frame, :, 2:].numpy()
        
        true_speed = np.linalg.norm(true_vel, axis=1)
        pred_speed = np.linalg.norm(pred_vel, axis=1)
        
        scat1.set_array(true_speed)
        scat2.set_array(pred_speed)
        
        return scat1, scat2
    
    frames = list(range(0, len(true_pos), show_every))
    anim = animation.FuncAnimation(fig, animate, frames=frames, 
                                 interval=100, blit=False, repeat=True)
    
    plt.tight_layout()
    
    if save_path:
        anim.save(save_path, writer='pillow', fps=10)
        print(f"Animation saved to {save_path}")
    
    plt.show()
    return anim

def plot_rollout_errors(metrics, save_path=None):
    """Plot rollout error evolution over time"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    timesteps = range(len(metrics['position_mse']))
    
    # Position error
    ax1.plot(timesteps, metrics['position_mse'], 'b-', linewidth=2, label='Position MSE')
    ax1.set_xlabel('Timestep')
    ax1.set_ylabel('Position MSE')
    ax1.set_title('Position Error Over Time')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Velocity error
    ax2.plot(timesteps, metrics['velocity_mse'], 'r-', linewidth=2, label='Velocity MSE')
    ax2.set_xlabel('Timestep')
    ax2.set_ylabel('Velocity MSE')
    ax2.set_title('Velocity Error Over Time')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Error plot saved to {save_path}")
    
    plt.show()

def compare_trajectories_static(true_trajectory, pred_trajectory, timesteps_to_show=[0, 10, 25, 50],
                               bounds=(0.1, 0.9), save_path=None):
    """
    Create static comparison of trajectories at different timesteps
    """
    fig, axes = plt.subplots(2, len(timesteps_to_show), figsize=(4*len(timesteps_to_show), 8))
    
    true_pos = true_trajectory[:, :, :2].numpy()
    pred_pos = pred_trajectory[:, :, :2].numpy()
    
    min_bound, max_bound = bounds
    
    for i, t in enumerate(timesteps_to_show):
        if t >= len(true_pos):
            continue
            
        # Ground truth
        ax_true = axes[0, i]
        ax_true.scatter(true_pos[t, :, 0], true_pos[t, :, 1], s=30, alpha=0.7, c='blue')
        ax_true.add_patch(Rectangle((min_bound, min_bound), 
                                  max_bound - min_bound, max_bound - min_bound,
                                  fill=False, edgecolor='black', linewidth=2))
        ax_true.set_xlim(min_bound - 0.05, max_bound + 0.05)
        ax_true.set_ylim(min_bound - 0.05, max_bound + 0.05)
        ax_true.set_aspect('equal')
        ax_true.set_title(f'Ground Truth t={t}')
        
        # Prediction
        ax_pred = axes[1, i]
        ax_pred.scatter(pred_pos[t, :, 0], pred_pos[t, :, 1], s=30, alpha=0.7, c='red')
        ax_pred.add_patch(Rectangle((min_bound, min_bound), 
                                  max_bound - min_bound, max_bound - min_bound,
                                  fill=False, edgecolor='black', linewidth=2))
        ax_pred.set_xlim(min_bound - 0.05, max_bound + 0.05)
        ax_pred.set_ylim(min_bound - 0.05, max_bound + 0.05)
        ax_pred.set_aspect('equal')
        ax_pred.set_title(f'Prediction t={t}')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")
    
    plt.show()

def custom_collate_fn(batch):
    """Custom collate function to handle variable number of particles"""
    # Find the maximum number of particles in this batch
    max_particles = max(x.shape[1] for x, y in batch)
    
    batch_x = []
    batch_y = []
    
    for x, y in batch:
        seq_len, num_particles, features = x.shape
        
        # Pad with zeros if needed
        if num_particles < max_particles:
            pad_size = max_particles - num_particles
            x_pad = torch.zeros(seq_len, pad_size, features)
            y_pad = torch.zeros(seq_len, pad_size, features)
            
            x = torch.cat([x, x_pad], dim=1)
            y = torch.cat([y, y_pad], dim=1)
        
        batch_x.append(x)
        batch_y.append(y)
    
    return torch.stack(batch_x), torch.stack(batch_y)

def train_model(model, train_loader, val_loader, num_epochs=100, lr=1e-3, device='cuda'):
    """Training loop for the particle transformer"""
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    model.to(device)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        epoch_train_loss = 0
        num_batches = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            # Handle sequence dimension - process each timestep
            batch_size, seq_len, num_particles, features = batch_x.shape
            
            optimizer.zero_grad()
            total_loss = 0
            
            # Process each timestep in the sequence
            for t in range(seq_len):
                pred = model(batch_x[:, t])  # (batch, particles, 4)
                target = batch_y[:, t]       # (batch, particles, 4)
                
                loss_dict = physics_informed_loss(pred, target)
                total_loss += loss_dict['total_loss']
            
            total_loss = total_loss / seq_len
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_train_loss += total_loss.item()
            num_batches += 1
        
        avg_train_loss = epoch_train_loss / num_batches
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        epoch_val_loss = 0
        num_val_batches = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                batch_size, seq_len, num_particles, features = batch_x.shape
                
                total_val_loss = 0
                for t in range(seq_len):
                    pred = model(batch_x[:, t])
                    target = batch_y[:, t]
                    
                    loss_dict = physics_informed_loss(pred, target)
                    total_val_loss += loss_dict['total_loss']
                
                total_val_loss = total_val_loss / seq_len
                epoch_val_loss += total_val_loss.item()
                num_val_batches += 1
        
        avg_val_loss = epoch_val_loss / num_val_batches
        val_losses.append(avg_val_loss)
        
        scheduler.step(avg_val_loss)
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch:3d}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}')
            print(f'           Gravity = [{model.gravity[0].item():.3f}, {model.gravity[1].item():.3f}]')
    
    return train_losses, val_losses

# Example usage and testing
if __name__ == "__main__":
    # Load dataset
    dataset = ParticleDataset('sample_data/water_drop/combined_dataset.npz', sequence_length=5)
    
    # Split into train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=custom_collate_fn)
    
    # Initialize model
    model = PhysicsInformedParticleTransformer(
        d_model=128,
        n_heads=8,
        n_layers=3,
        dropout=0.1,
        gravity=0.0,
        bounds=(0.1, 0.9),
        dt=0.01
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on device: {device}")
    
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, 
        num_epochs=50, lr=1e-3, device=device
    )
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Progress')
    plt.yscale('log')
    plt.grid(True)
    plt.show()
    
    # Save model
    torch.save(model.state_dict(), 'particle_transformer.pth')
    print("Model saved as 'particle_transformer.pth'")
    
    # === EVALUATION AND VISUALIZATION ===
    print("\n=== Evaluating Model Performance ===")
    
    # Evaluate rollout accuracy
    metrics = evaluate_rollout_accuracy(model, dataset, num_rollouts=3, rollout_length=50, device=device)
    print(f"Final Position MSE: {metrics['final_position_mse']:.6f}")
    print(f"Final Velocity MSE: {metrics['final_velocity_mse']:.6f}")
    
    # Plot error evolution
    plot_rollout_errors(metrics, save_path='rollout_errors.png')
    
    # Visualize a specific trajectory
    print("\n=== Creating Visualizations ===")
    
    # Get a test trajectory
    test_trajectory = dataset.get_full_trajectory(0)
    initial_state = test_trajectory[0:1]
    
    # Generate prediction
    pred_trajectory = model.multi_step_rollout(initial_state, 100, device=device)
    true_trajectory = test_trajectory[:101]  # Match length
    
    # Static comparison
    compare_trajectories_static(true_trajectory, pred_trajectory, 
                              timesteps_to_show=[0, 25, 50, 75],
                              save_path='trajectory_comparison.png')
    
    # Animated visualization (comment out if running in non-interactive environment)
    print("Creating animation...")
    anim = visualize_particle_trajectory(true_trajectory, pred_trajectory, 
                                       save_path='particle_animation.gif', 
                                       show_every=2)
    
    print("\nEvaluation complete! Check the generated plots and animation.")