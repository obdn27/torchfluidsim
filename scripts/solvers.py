import torch
import numpy as np

import torch

def interaction_step(frame, interaction_radius, mouse_x, mouse_y, grid_resolution, window_res, mouse_acceleration, dt, decay_rate):
    """
    Implements the AddMouseVelocityAndAdvection kernel in PyTorch.

    Args:
    - frame (Tensor): The simulation state tensor of shape (width, height, 3).
      (Density, X-Velocity, Y-Velocity).
    - interaction_radius (float): The radius of interaction.
    - mouse_x (float): The X-coordinate of the mouse in window space.
    - mouse_y (float): The Y-coordinate of the mouse in window space.
    - grid_resolution (tuple): The resolution of the simulation grid.
    - window_res (tuple): The resolution of the display window.
    - mouse_acceleration (tuple): The acceleration applied by the mouse (dx, dy).
    - dt (float): Simulation timestep.
    - decay_rate (float): The decay rate to apply to the density values to prevent explosion.

    Returns:
    - Updated frame tensor.
    """

    # Normalize mouse position to grid coordinates
    norm_mouse_x = (mouse_x / window_res[0]) * grid_resolution[0]
    norm_mouse_y = (mouse_y / window_res[1]) * grid_resolution[1]

    height, width = grid_resolution

    y, x = torch.meshgrid(
        torch.arange(height, dtype=torch.float32, device=frame.device),
        torch.arange(width, dtype=torch.float32, device=frame.device),
        indexing='ij'
    )

    distance = torch.sqrt((x - norm_mouse_x) ** 2 + (y - norm_mouse_y) ** 2)
    mask = distance < interaction_radius  # Boolean mask

    falloff = (1.0 - (distance / interaction_radius)).clamp(min=0)
    falloff = falloff * mask.float()  # Ensure zero outside radius

    density, x_vel, y_vel = frame[:, :, 0], frame[:, :, 1], frame[:, :, 2]

    x_vel = x_vel + (mouse_acceleration[0] * falloff)
    y_vel = y_vel + (mouse_acceleration[1] * falloff)

    density += (100.0 * falloff * dt)

    density *= decay_rate

    updated_frame = torch.stack([density, x_vel, y_vel], dim=-1)

    return updated_frame


def bilinear_interpolation(field, x, y):
    """
    Performs bilinear interpolation for a given field at fractional coordinates (x, y).
    
    Args:
    - field (torch.Tensor): Input field of shape (H, W).
    - x (torch.Tensor): X coordinates (can be fractional).
    - y (torch.Tensor): Y coordinates (can be fractional).

    Returns:
    - torch.Tensor: Interpolated values at the given (x, y) positions.
    """

    H, W = field.shape

    # Get integer and fractional parts of coordinates
    x0 = torch.floor(x).long()
    x1 = x0 + 1
    y0 = torch.floor(y).long()
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, W - 1)
    x1 = torch.clamp(x1, 0, W - 1)
    y0 = torch.clamp(y0, 0, H - 1)
    y1 = torch.clamp(y1, 0, H - 1)

    # Extract field values at four neighboring grid points
    Q11 = field[y0, x0]  # Top-left
    Q21 = field[y0, x1]  # Top-right
    Q12 = field[y1, x0]  # Bottom-left
    Q22 = field[y1, x1]  # Bottom-right

    # Compute interpolation weights
    wx = x - x0.float()
    wy = y - y0.float()

    # Bilinear interpolation formula
    top = (1 - wx) * Q11 + wx * Q21
    bottom = (1 - wx) * Q12 + wx * Q22
    interpolated = (1 - wy) * top + wy * bottom

    return interpolated


def advection_step(frame, dt, grid_resolution):
    height, width = grid_resolution

    posy, posx = torch.meshgrid(
        torch.arange(height, dtype=torch.float32, device=frame.device),
        torch.arange(width, dtype=torch.float32, device=frame.device),
        indexing='ij',
    )

    density, x_vel, y_vel = frame[:, :, 0], frame[:, :, 1], frame[:, :, 2]

    backtracked_x = posx - dt * x_vel
    backtracked_y = posy - dt * y_vel
    backtracked_x = torch.clamp(backtracked_x, 0.5, width - 1.5)
    backtracked_y = torch.clamp(backtracked_y, 0.5, height - 1.5)

    advected_density = bilinear_interpolation(density, backtracked_x, backtracked_y)
    advected_x_vel = bilinear_interpolation(x_vel, backtracked_x, backtracked_y)
    advected_y_vel = bilinear_interpolation(y_vel, backtracked_x, backtracked_y)

    advected_frame = torch.stack([advected_density, advected_x_vel, advected_y_vel], dim=-1)

    return advected_frame


