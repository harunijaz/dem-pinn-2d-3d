# DEM-PINN: Deep Energy Method with Physics-Informed Neural Networks for 3D Voxel Data

## Overview

This repository contains the implementation of a Physics-Informed Neural Network (PINN) combined with the Deep Energy Method (DEM) to solve 3D elasticity problems using voxel data. The code is designed to predict displacements and stresses in a 3D voxelized structure, such as a chair, under given force conditions. The model leverages the principles of elasticity and energy minimization to ensure that the predictions are physically consistent.

## Key Features

- **Physics-Informed Neural Networks (PINNs):** The model integrates physical laws (elasticity equations) directly into the neural network training process, ensuring that the predictions adhere to the underlying physics.
- **Deep Energy Method (DEM):** The energy-based approach is used to formulate the loss function, which includes terms for internal energy, external work, and boundary conditions.
- **3D Voxel Data:** The model is designed to work with 3D voxel data, making it suitable for applications in structural analysis, material science, and computer graphics.
- **Elasticity Parameters:** The model incorporates Young's modulus and Poisson's ratio to simulate material properties accurately.
- **Customizable Neural Network:** The neural network architecture is flexible and can be adjusted to include more layers or neurons as needed.

## Requirements

- Python 3.x
- PyTorch
- NumPy
- Matplotlib
- SciPy

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/dem-pinn-3d.git
   cd dem-pinn-3d
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Prepare Voxel Data:**
   Ensure that your voxel data is stored in a `.mat` file with the key `'voxels3D'`. The data should represent a 3D structure, such as a chair.

2. **Set Parameters:**
   Modify the parameters in the script to match your problem setup. Key parameters include:
   - `voxel_file`: Path to the `.mat` file containing the voxel data.
   - `E`: Young's modulus of the material.
   - `nu`: Poisson's ratio of the material.
   - `force_magnitude`: Magnitude of the applied force.

3. **Train the Model:**
   Run the script to train the model:
   ```bash
   python dem-pinn-3d.py
   ```

4. **Visualize Results:**
   After training, the script will save the trained model and plot the loss history. You can also visualize the predicted displacements and stresses.

## Code Structure

- **FFNN Class:** Defines the feedforward neural network architecture.
- **VoxelChairPINN Class:** Implements the PINN with DEM for 3D voxel data.
  - `strain_tensor`: Computes the strain tensor.
  - `stress_tensor`: Computes the stress tensor using the constitutive equation.
  - `strain_energy_density`: Computes the strain energy density.
  - `internal_energy`: Computes the internal energy of the system.
  - `external_work`: Computes the external work done by applied forces.
  - `boundary_condition_loss`: Enforces boundary conditions.
  - `dem_pinn_loss`: Combines internal energy, external work, and boundary conditions into the loss function.
  - `train`: Trains the model using the Adam optimizer.
  - `get_displacement`: Retrieves the predicted displacements.
  - `visualize_loss`: Plots the training loss history.

## Example

```python
# Set up parameters
voxel_file = '/path/to/voxel_data.mat'
E = 1e9  # Young's modulus (Pa)
nu = 0.3  # Poisson's ratio
force_magnitude = 1.0  # Force magnitude

# Create and train the model
voxel_pinn = VoxelChairPINN(voxel_file, E, nu, force_magnitude)
voxel_pinn.train(num_epochs=100)

# Visualize loss function after training
voxel_pinn.visualize_loss()

# Save the trained model
torch.save(voxel_pinn.model.state_dict(), "trained_voxel_chair_model.pth")
```

## Results

After training, the model will output the predicted displacements and stresses for the given voxel structure. The loss history plot will help you assess the training process and convergence.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any bugs or feature requests.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- This project is inspired by the work on Physics-Informed Neural Networks (PINNs) and the Deep Energy Method (DEM).
- Special thanks to the authors of the original papers and the open-source community for their contributions to the field.

## Contact

For any questions or feedback, please contact [Your Name] at [your.email@example.com].
