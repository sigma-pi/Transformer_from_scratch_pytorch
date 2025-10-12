import numpy as np
import os

def combine_npz_files(input_files, output_file):
  """
  Combine multiple NPZ files containing trajectories into a single NPZ file.
  
  Args:
      input_files: List of input NPZ file paths
      output_file: Output NPZ file path
  """
  all_trajectories = {}
  trajectory_counter = 0
  
  print("Loading and combining trajectories...")
  
  for file_path in input_files:
      print(f"\nProcessing: {file_path}")
      
      # Load the NPZ file with pickle enabled
      data = np.load(file_path, allow_pickle=True)
      
      # Get all trajectory keys from this file
      trajectory_keys = [key for key in data.keys() if key.startswith('simulation_trajectory_')]
      
      print(f"Found {len(trajectory_keys)} trajectories: {trajectory_keys}")
      
      # Add each trajectory to the combined dataset with new numbering
      for key in trajectory_keys:
          new_key = f"simulation_trajectory_{trajectory_counter}"
          all_trajectories[new_key] = data[key]
          print(f"  {key} -> {new_key}")
          trajectory_counter += 1
      
      # Close the data file
      data.close()
  
  # Save the combined dataset
  print(f"\nSaving combined dataset to: {output_file}")
  print(f"Total trajectories: {trajectory_counter}")
  
  # Create output directory if it doesn't exist (only if there's a directory path)
  output_dir = os.path.dirname(output_file)
  if output_dir:  # Only create directory if there's actually a directory path
      os.makedirs(output_dir, exist_ok=True)
  
  # Save as NPZ file
  np.savez_compressed(output_file, **all_trajectories)
  
  print("✅ Successfully combined all trajectories!")
  
  # Verify the combined file
  verify_combined_file(output_file)

def verify_combined_file(file_path):
  """Verify the combined NPZ file and show summary information."""
  print(f"\n📊 Verifying combined file: {file_path}")
  
  data = np.load(file_path, allow_pickle=True)
  trajectory_keys = [key for key in data.keys() if key.startswith('simulation_trajectory_')]
  
  print(f"Total trajectories in combined file: {len(trajectory_keys)}")
  
  for key in sorted(trajectory_keys):
      trajectory = data[key]
      print(f"  {key}: shape {trajectory.shape}")
      
      # Show some basic stats for the first trajectory
      if key == 'simulation_trajectory_0':
          print(f"    Data type: {trajectory.dtype}")
          # Since these are object arrays, let's inspect the contents
          print(f"    Content preview: {trajectory}")
          if len(trajectory) > 0:
              print(f"    First element type: {type(trajectory[0])}")
              if hasattr(trajectory[0], 'shape'):
                  print(f"    First element shape: {trajectory[0].shape}")
  
  data.close()

def analyze_original_files(input_files):
  """Analyze the original files to understand their structure."""
  print("🔍 Analyzing original files...")
  
  for file_path in input_files:
      print(f"\n📁 {file_path}:")
      data = np.load(file_path, allow_pickle=True)
      
      trajectory_keys = [key for key in data.keys() if key.startswith('simulation_trajectory_')]
      print(f"  Trajectories: {len(trajectory_keys)}")
      
      for key in trajectory_keys:
          trajectory = data[key]
          print(f"    {key}: {trajectory.shape} ({trajectory.dtype})")
          
          # Since these are object arrays, let's peek at the content
          if key == 'simulation_trajectory_0':
              print(f"      Content preview: {trajectory}")
              if len(trajectory) > 0:
                  print(f"      First element type: {type(trajectory[0])}")
                  if hasattr(trajectory[0], 'shape'):
                      print(f"      First element shape: {trajectory[0].shape}")
      
      data.close()

if __name__ == "__main__":
  # Input files
  input_files = [
      "test.npz",
      "train.npz", 
      "valid.npz"
  ]
  
  # Output file
  output_file = "combined_dataset.npz"
  
  # First, analyze the original files
  analyze_original_files(input_files)
  
  # Combine all trajectories
  combine_npz_files(input_files, output_file)
  
  print(f"\n🎉 All done! Your combined dataset is saved as: {output_file}")