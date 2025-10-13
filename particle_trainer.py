import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import numpy as np
from particle_models import ParticleDataset, PhysicsInformedParticleTransformer

def physics_informed_loss(pred, target, gravity_weight=0.01, bounds=(0.1, 0.9), dt=0.01):
    """Combined loss function with physics constraints"""
    
    # Extract positions and velocities
    pred_pos, pred_vel = pred[:, :, :2], pred[:, :, 2:]
    true_pos, true_vel = target[:, :, :2], target[:, :, 2:]
    
    # Standard MSE losses
    pos_loss = F.mse_loss(pred_pos, true_pos)
    vel_loss = F.mse_loss(pred_vel, true_vel)
    
    # Softer boundary violation penalty
    min_bound, max_bound = bounds
    margin = 0.02
    
    # Only penalize significant violations
    violation_min = torch.relu(min_bound - pred_pos)
    violation_max = torch.relu(pred_pos - max_bound)
    boundary_loss = (violation_min + violation_max).mean()
    
    # Combine losses
    total_loss = pos_loss * 100 + vel_loss * 10 + boundary_loss * 0.01
    
    return {
        'total_loss': total_loss,
        'pos_loss': pos_loss,
        'vel_loss': vel_loss,
        'boundary_loss': boundary_loss
    }

def evaluate_rollout_accuracy(model, dataset, num_rollouts=5, rollout_length=50, device='cpu'):
    """
    Evaluate model accuracy on multi-step rollouts
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
                                 save_path='particle_animation.gif', show_every=2):
    """
    Create an animated visualization of particle trajectories
    """
    # Set matplotlib backend for headless operation
    plt.switch_backend('Agg')
    
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
    
    def animate(frame_idx):
        frame = frames[frame_idx] if frame_idx < len(frames) else frames[-1]
        if frame >= len(true_pos):
            frame = len(true_pos) - 1
            
        # Update scatter plots
        scat1.set_offsets(true_pos[frame])
        scat2.set_offsets(pred_pos[frame])
        
        # Color particles by velocity magnitude for visual appeal
        if frame < len(true_trajectory):
            true_vel = true_trajectory[frame, :, 2:].numpy()
            pred_vel = pred_trajectory[frame, :, 2:].numpy()
            
            true_speed = np.linalg.norm(true_vel, axis=1)
            pred_speed = np.linalg.norm(pred_vel, axis=1)
            
            scat1.set_array(true_speed)
            scat2.set_array(pred_speed)
        
        return scat1, scat2
    
    frames = list(range(0, min(len(true_pos), len(pred_pos)), show_every))
    
    try:
        anim = animation.FuncAnimation(fig, animate, frames=len(frames), 
                                     interval=200, blit=False, repeat=True)
        
        plt.tight_layout()
        
        # Try different writers
        try:
            anim.save(save_path, writer='pillow', fps=5)
            print(f"Animation saved to {save_path}")
        except Exception as e:
            print(f"Failed to save with pillow: {e}")
            try:
                anim.save(save_path, writer='imagemagick', fps=5)
                print(f"Animation saved to {save_path} (using imagemagick)")
            except Exception as e2:
                print(f"Failed to save animation: {e2}")
                print("Skipping animation creation")
        
    except Exception as e:
        print(f"Error creating animation: {e}")
    
    plt.close(fig)  # Close figure to free memory

def plot_rollout_errors(metrics, save_path='rollout_errors.png'):
    """Plot rollout error evolution over time"""
    plt.switch_backend('Agg')
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
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Error plot saved to {save_path}")
    plt.close(fig)  # Close figure to free memory

def compare_trajectories_static(true_trajectory, pred_trajectory, timesteps_to_show=[0, 10, 25, 50],
                               bounds=(0.1, 0.9), save_path='trajectory_comparison.png'):
    """
    Create static comparison of trajectories at different timesteps
    """
    plt.switch_backend('Agg')
    fig, axes = plt.subplots(2, len(timesteps_to_show), figsize=(4*len(timesteps_to_show), 8))
    
    true_pos = true_trajectory[:, :, :2].numpy()
    pred_pos = pred_trajectory[:, :, :2].numpy()
    
    min_bound, max_bound = bounds
    
    for i, t in enumerate(timesteps_to_show):
        if t >= len(true_pos) or t >= len(pred_pos):
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
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to {save_path}")
    plt.close(fig)  # Close figure to free memory

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
    
    # Separate optimizers for different parts
    main_params = [p for name, p in model.named_parameters() if 'gravity' not in name]
    gravity_params = [p for name, p in model.named_parameters() if 'gravity' in name]
    
    optimizer = torch.optim.Adam([
        {'params': main_params, 'lr': lr},
        {'params': gravity_params, 'lr': lr * 0.1}  # Slower learning for gravity
    ], weight_decay=1e-5)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)
    
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
        
        if epoch % 5 == 0:
            print(f'Epoch {epoch:3d}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}, Gravity = [{model.gravity[0].item():.4f}, {model.gravity[1].item():.4f}]')
    
    return train_losses, val_losses

# Main training and evaluation script
if __name__ == "__main__":
    # Set matplotlib backend for headless operation
    plt.switch_backend('Agg')
    
    # Load dataset
    dataset = ParticleDataset('sample_data/water_drop/single_trajectory.npz', sequence_length=20)
    
    # Split into train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=custom_collate_fn)
    
    # Initialize model with small gravity
    model = PhysicsInformedParticleTransformer(
        d_model=128,
        n_heads=8,
        n_layers=3,
        dropout=0.01,
        gravity=0.005,  # Small gravity
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
    
    # Plot training curves and save
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Progress')
    plt.yscale('log')
    plt.grid(True)
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    print("Training curves saved to 'training_curves.png'")
    plt.close()
    
    # Save model
    torch.save(model.state_dict(), 'particle_transformer.pth')
    print("Model saved as 'particle_transformer.pth'")
    
    # === EVALUATION AND VISUALIZATION ===
    print("\n=== Evaluating Model Performance ===")
    
    # Evaluate rollout accuracy
    metrics = evaluate_rollout_accuracy(model, dataset, num_rollouts=3, rollout_length=50, device=device)
    
    print(f"Final Position MSE: {metrics['final_position_mse']:.6f}")
    print(f"Final Velocity MSE: {metrics['final_velocity_mse']:.6f}")
    
    # Plot rollout errors
    plot_rollout_errors(metrics, save_path='rollout_errors.png')
    
    print("\n=== Creating Visualizations ===")
    
    # Get a sample trajectory for visualization
    sample_traj = dataset.get_full_trajectory(0)
    initial_state = sample_traj[0:1]
    
    # Generate prediction for visualization (shorter rollout for animation)
    pred_traj = model.multi_step_rollout(initial_state, 100, device=device)
    true_traj = sample_traj[:101]  # Match length
    
    # Create static comparison
    compare_trajectories_static(true_traj, pred_traj, 
                              timesteps_to_show=[0, 20, 50, 100],
                              save_path='trajectory_comparison.png')
    
    # Create animated visualization (if possible)
    try:
        visualize_particle_trajectory(true_traj, pred_traj, 
                                    save_path='particle_animation.gif',
                                    show_every=3)
    except Exception as e:
        print(f"Animation creation failed: {e}")
        print("Static comparison created instead")
    
    print("\n=== Summary ===")
    print(f"Training completed with final validation loss: {val_losses[-1]:.6f}")
    print(f"Model saved as 'particle_transformer.pth'")
    print(f"Learned gravity: [{model.gravity[0].item():.4f}, {model.gravity[1].item():.4f}]")
    print("Visualizations saved:")
    print("  - training_curves.png")
    print("  - rollout_errors.png") 
    print("  - trajectory_comparison.png")
    print("  - particle_animation.gif (if successful)")
    
    print("\nTo use the trained model:")
    print("from particle_models import PhysicsInformedParticleTransformer")
    print("model = PhysicsInformedParticleTransformer()")
    print("model.load_state_dict(torch.load('particle_transformer.pth'))")