# NERF
NERF Modeling Project - 

2dfittingScene fits a 2D image using an MLP that maps (x, y) coordinates to RGB values, with positional encoding at 0, 2, and 6 frequencies.

NERF reconstructs a 3D scene from multi-view images using:
- Ray generation from camera intrinsics and poses
- Stratified sampling along rays
- A NeRF-style MLP with skip connections
- Volumetric rendering to compute pixel colors
- A full training loop to render novel views

- Trained models and data provided
