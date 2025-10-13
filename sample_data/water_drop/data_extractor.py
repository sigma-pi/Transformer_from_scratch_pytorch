import numpy as np

def extract_first_trajectory(input_file, output_file):
  """
  Extract the first trajectory from an NPZ file and save it to a new NPZ file.
  
  Args:
      input_file: Path to the input NPZ file
      output_file: Path to the output NPZ file
  """
  print(f"Loading NPZ file: {input_file}")
  
  # Load the NPZ file with pickle enabled
  data = np.load(input_file, allow_pickle=True)
  
  # Get all trajectory keys
  trajectory_keys = [key for key in data.keys() if key.startswith('simulation_trajectory_')]
  trajectory_keys.sort()  # Sort to ensure consistent ordering
  
  print(f"Found {len(trajectory_keys)} trajectories: {trajectory_keys}")
  
  if len(trajectory_keys) == 0:
      print("❌ No trajectories found in the file!")
      data.close()
      return
  
  # Get the first trajectory
  first_key = trajectory_keys[0]
  first_trajectory = data[first_key]
  
  print(f"Extracting trajectory: {first_key}")
  print(f"  Shape: {first_trajectory.shape}")
  print(f"  Data type: {first_trajectory.dtype}")
  
  # If it's an object array, let's see what's inside
  if first_trajectory.dtype == 'object':
      print(f"  Content preview: {first_trajectory}")
      if len(first_trajectory) > 0:
          print(f"  First element type: {type(first_trajectory[0])}")
          if hasattr(first_trajectory[0], 'shape'):
              print(f"  First element shape: {first_trajectory[0].shape}")
  
  # Close the original data
  data.close()
  
  # Save the first trajectory to a new NPZ file
  print(f"\nSaving first trajectory to: {output_file}")
  
  # Save with the same key name
  np.savez_compressed(output_file, **{first_key: first_trajectory})
  
  print("✅ Successfully extracted first trajectory!")
  
  # Verify the new file
  verify_extracted_file(output_file)

def verify_extracted_file(file_path):
  """Verify the extracted NPZ file."""
  print(f"\n📊 Verifying extracted file: {file_path}")
  
  data = np.load(file_path, allow_pickle=True)
  trajectory_keys = [key for key in data.keys() if key.startswith('simulation_trajectory_')]
  
  print(f"Trajectories in extracted file: {len(trajectory_keys)}")
  
  for key in trajectory_keys:
      trajectory = data[key]
      print(f"  {key}: shape {trajectory.shape}, dtype {trajectory.dtype}")
      
      if trajectory.dtype == 'object':
          print(f"    Content: {trajectory}")
  
  data.close()

if __name__ == "__main__":
  # Input and output files
  input_file = "test.npz"
  output_file = "single_trajectory.npz"
  
  # Extract the first trajectory
  extract_first_trajectory(input_file, output_file)
  
  print(f"\n🎉 Done! First trajectory saved as: {output_file}")