import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
import os
from pathlib import Path

def load_simulation_data(npz_file_path):
    """
    Load simulation data from npz file
    
    Args:
        npz_file_path (str): Path to the npz file
        
    Returns:
        dict: Dictionary containing all simulation trajectories
    """
    # Load with allow_pickle=True to handle object arrays
    data = np.load(npz_file_path, allow_pickle=True)
    trajectories = {}
    
    print(f"Available keys in the file: {list(data.keys())}")
    
    # Find all simulation trajectory keys
    for key in data.keys():
        if key.startswith('simulation_trajectory_'):
            try:
                traj_data = data[key]
                print(f"Loading {key}: type={type(traj_data)}, shape={getattr(traj_data, 'shape', 'N/A')}")
                
                # Handle different data structures
                if isinstance(traj_data, np.ndarray) and traj_data.dtype == object:
                    # If it's an object array, extract the contents
                    if len(traj_data) >= 2:
                        positions = traj_data[0]
                        materials = traj_data[1]
                        print(f"  Positions shape: {positions.shape}")
                        print(f"  Materials shape: {materials.shape}")
                        trajectories[key] = (positions, materials)
                    else:
                        print(f"  Warning: {key} has unexpected structure")
                else:
                    # Direct array access
                    trajectories[key] = traj_data
                    
            except Exception as e:
                print(f"Error loading {key}: {e}")
                continue
                
    return trajectories

def create_water_drop_animation(positions, materials, trajectory_name, output_dir, 
                              figsize=(10, 8), dpi=100, fps=30, particle_size=20):
    """
    Create animated GIF of water drop simulation
    
    Args:
        positions (np.array): Array of shape (timesteps, particles, 2)
        materials (np.array): Array of shape (particles,) with material types
        trajectory_name (str): Name of the trajectory for file naming
        output_dir (str): Output directory for saving GIF
        figsize (tuple): Figure size for the plot
        dpi (int): DPI for the output
        fps (int): Frames per second for the animation
        particle_size (int): Size of particles in the plot
    """
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Ensure positions is a numpy array with correct shape
    positions = np.array(positions)
    if len(positions.shape) != 3 or positions.shape[2] != 2:
        print(f"Error: Expected positions shape (timesteps, particles, 2), got {positions.shape}")
        return
    
    # Get data dimensions
    n_timesteps, n_particles, _ = positions.shape
    print(f"Creating animation with {n_timesteps} timesteps and {n_particles} particles")
    
    # Calculate plot boundaries with some padding
    x_min, x_max = positions[:, :, 0].min(), positions[:, :, 0].max()
    y_min, y_max = positions[:, :, 1].min(), positions[:, :, 1].max()
    
    # Add padding (10% of range)
    x_range = x_max - x_min
    y_range = y_max - y_min
    padding_x = max(x_range * 0.1, 0.1)  # Minimum padding
    padding_y = max(y_range * 0.1, 0.1)
    
    x_min -= padding_x
    x_max += padding_x
    y_min -= padding_y
    y_max += padding_y
    
    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')
    ax.set_title(f'Water Drop Simulation - {trajectory_name}', fontsize=14, fontweight='bold')
    ax.set_xlabel('X Position', fontsize=12)
    ax.set_ylabel('Y Position', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Create color map for water particles (blue theme)
    water_color = '#1f77b4'  # Blue for water particles
    
    # Initialize scatter plot
    scat = ax.scatter([], [], s=particle_size, c=water_color, alpha=0.7, 
                     edgecolors='darkblue', linewidth=0.5)
    
    # Add time text
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=12,
                       verticalalignment='top', 
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Add particle count text
    particle_text = ax.text(0.02, 0.92, f'Particles: {n_particles}', transform=ax.transAxes, 
                          fontsize=10, verticalalignment='top', 
                          bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Add material info
    unique_materials = np.unique(materials)
    material_text = ax.text(0.02, 0.86, f'Materials: {unique_materials}', transform=ax.transAxes, 
                          fontsize=10, verticalalignment='top', 
                          bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    def animate(frame):
        """Animation function for each frame"""
        # Get current positions
        current_positions = positions[frame]
        
        # Update scatter plot data
        scat.set_offsets(current_positions)
        
        # Update time text
        time_text.set_text(f'Time Step: {frame}/{n_timesteps-1}')
        
        return scat, time_text
    
    # Create animation
    print(f"Creating animation for {trajectory_name}...")
    anim = animation.FuncAnimation(fig, animate, frames=n_timesteps, 
                                 interval=1000/fps, blit=True, repeat=True)
    
    # Save as GIF with dataset prefix
    dataset_name = os.path.basename(output_dir)  # Extract dataset name from output_dir
    output_path = os.path.join(output_dir, f'{dataset_name}_{trajectory_name}.gif')
    print(f"Saving animation to {output_path}...")
    
    try:
        # Use PillowWriter for better GIF quality
        writer = animation.PillowWriter(fps=fps)
        anim.save(output_path, writer=writer)
        print(f"Animation saved successfully!")
    except Exception as e:
        print(f"Error saving animation: {e}")
        # Try alternative method
        try:
            anim.save(output_path, writer='pillow', fps=fps)
            print(f"Animation saved with alternative method!")
        except Exception as e2:
            print(f"Failed to save animation: {e2}")
    
    plt.close(fig)

def visualize_all_trajectories(npz_file_path, output_dir='./simulation', 
                              figsize=(10, 8), dpi=100, fps=20, particle_size=15):
    """
    Visualize all trajectories in the npz file
    
    Args:
        npz_file_path (str): Path to the npz file
        output_dir (str): Output directory for saving GIFs
        figsize (tuple): Figure size for plots
        dpi (int): DPI for output
        fps (int): Frames per second for animations
        particle_size (int): Size of particles in plots
    """
    
    print(f"Loading data from {npz_file_path}...")
    
    # Check if file exists
    if not os.path.exists(npz_file_path):
        print(f"Error: File {npz_file_path} not found!")
        return
    
    trajectories = load_simulation_data(npz_file_path)
    
    if not trajectories:
        print("No simulation trajectories found in the file!")
        return
    
    print(f"Found {len(trajectories)} trajectories to visualize")
    
    for traj_name, traj_data in trajectories.items():
        print(f"\nProcessing {traj_name}...")
        
        try:
            # Handle different data structures
            if isinstance(traj_data, tuple) and len(traj_data) >= 2:
                positions = traj_data[0]
                materials = traj_data[1]
            elif isinstance(traj_data, np.ndarray) and traj_data.dtype == object:
                positions = traj_data[0]
                materials = traj_data[1]
            else:
                print(f"  Warning: Unexpected data structure for {traj_name}")
                continue
            
            # Convert to numpy arrays if needed
            positions = np.array(positions)
            materials = np.array(materials)
            
            print(f"  - Timesteps: {positions.shape[0]}")
            print(f"  - Particles: {positions.shape[1]}")
            print(f"  - Dimensions: {positions.shape[2]}")
            print(f"  - Unique materials: {np.unique(materials)}")
            print(f"  - Position range: X[{positions[:,:,0].min():.2f}, {positions[:,:,0].max():.2f}], Y[{positions[:,:,1].min():.2f}, {positions[:,:,1].max():.2f}]")
            
            # Create animation
            create_water_drop_animation(positions, materials, traj_name, output_dir,
                                      figsize=figsize, dpi=dpi, fps=fps, particle_size=particle_size)
            
        except Exception as e:
            print(f"  Error processing {traj_name}: {e}")
            continue
    
    print(f"\nAll animations completed! Check the '{output_dir}' folder for GIF files.")

def create_summary_plot(npz_file_path, output_dir='./simulation'):
    """
    Create a summary plot showing initial and final states of all trajectories
    
    Args:
        npz_file_path (str): Path to the npz file
        output_dir (str): Output directory for saving plots
    """
    
    trajectories = load_simulation_data(npz_file_path)
    n_trajectories = len(trajectories)
    
    if n_trajectories == 0:
        return
    
    # Create subplots
    fig, axes = plt.subplots(2, n_trajectories, figsize=(4*n_trajectories, 8))
    if n_trajectories == 1:
        axes = axes.reshape(2, 1)
    
    for i, (traj_name, traj_data) in enumerate(trajectories.items()):
        try:
            # Handle data structure
            if isinstance(traj_data, tuple):
                positions = traj_data[0]
            else:
                positions = traj_data[0]
            
            positions = np.array(positions)
            
            # Initial state (top row)
            ax_init = axes[0, i]
            ax_init.scatter(positions[0, :, 0], positions[0, :, 1], 
                           c='#1f77b4', s=10, alpha=0.7)
            ax_init.set_title(f'{traj_name}\nInitial State')
            ax_init.set_aspect('equal')
            ax_init.grid(True, alpha=0.3)
            
            # Final state (bottom row)
            ax_final = axes[1, i]
            ax_final.scatter(positions[-1, :, 0], positions[-1, :, 1], 
                            c='#1f77b4', s=10, alpha=0.7)
            ax_final.set_title(f'Final State')
            ax_final.set_aspect('equal')
            ax_final.grid(True, alpha=0.3)
            
        except Exception as e:
            print(f"Error creating summary for {traj_name}: {e}")
            continue
    
    plt.tight_layout()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    dataset_name = os.path.basename(output_dir)
    summary_path = os.path.join(output_dir, f'{dataset_name}_trajectory_summary.png')
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Summary plot saved to {summary_path}")

# Main execution
if __name__ == "__main__":
    # Configuration
    DATASET_NAME = "valid" 
    NPZ_FILE_PATH = f"./sample_data/water_drop/{DATASET_NAME}.npz"
    OUTPUT_DIR = f"./simulation/water_drop/{DATASET_NAME}"
    
    # Animation settings
    FIGURE_SIZE = (12, 9)
    DPI = 100
    FPS = 25  # Frames per second
    PARTICLE_SIZE = 20  # Size of particles in the plot
    
    # Create visualizations
    visualize_all_trajectories(NPZ_FILE_PATH, OUTPUT_DIR, 
                              figsize=FIGURE_SIZE, dpi=DPI, 
                              fps=FPS, particle_size=PARTICLE_SIZE)
    
    # Create summary plot
    create_summary_plot(NPZ_FILE_PATH, OUTPUT_DIR)
    
    print("\nVisualization complete!")
    print(f"Check the '{OUTPUT_DIR}' folder for:")
    print("- Individual trajectory GIF files")
    print("- Summary plot showing initial and final states")
