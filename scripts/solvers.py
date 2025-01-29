import torch
import numpy as np
import torch.nn.functional as F


def interaction_step(frame, interaction_radius, interaction_strength, mouse_x, mouse_y, grid_resolution, window_res, mouse_acceleration, dt, decay_rate):
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

    density, x_vel, y_vel, divergence, pressure = frame[:, :, 0], frame[:, :, 1], frame[:, :, 2], frame[:, :, 3], frame[:, :, 4]

    x_vel = x_vel + (mouse_acceleration[0] * falloff)
    y_vel = y_vel + (mouse_acceleration[1] * falloff)

    density += (100.0 * falloff * dt) * interaction_strength

    updated_frame = torch.stack([density, x_vel, y_vel, divergence, pressure], dim=-1)

    return updated_frame * decay_rate


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

    density, x_vel, y_vel, divergence, pressure = frame[:, :, 0], frame[:, :, 1], frame[:, :, 2], frame[:, :, 3], frame[:, :, 4]

    # Compute backtracked positions
    backtracked_x = posx - dt * x_vel
    backtracked_y = posy - dt * y_vel
    backtracked_x = torch.clamp(backtracked_x / (width - 1) * 2 - 1, -1, 1)
    backtracked_y = torch.clamp(backtracked_y / (height - 1) * 2 - 1, -1, 1)

    # Stack grid positions into a format suitable for grid_sample
    grid = torch.stack([backtracked_x, backtracked_y], dim=-1).unsqueeze(0)  # (1, H, W, 2)

    # Perform interpolation
    advected_density = F.grid_sample(density.unsqueeze(0).unsqueeze(0), grid, mode='bilinear', align_corners=True).squeeze()
    advected_x_vel = F.grid_sample(x_vel.unsqueeze(0).unsqueeze(0), grid, mode='bilinear', align_corners=True).squeeze()
    advected_y_vel = F.grid_sample(y_vel.unsqueeze(0).unsqueeze(0), grid, mode='bilinear', align_corners=True).squeeze()

    advected_frame = torch.stack([advected_density, advected_x_vel, advected_y_vel, divergence, pressure], dim=-1)

    return advected_frame


def diffuse_step(frame, viscosity, diffusion_coeff, dt, iterations=20):
    """
    Uses Gauss-Seidel iterations to diffuse velocity and density fields.
    """

    for _ in range(int(int(iterations))):
        up = torch.roll(frame, -1, dims=0)
        down = torch.roll(frame, 1, dims=0)
        left = torch.roll(frame, -1, dims=1)
        right = torch.roll(frame, 1, dims=1)

        avg = (up + down + left + right) / 4

        frame[:, :, 0] += (avg[:, :, 0] - frame[:, :, 0]) * (1 - torch.exp(-diffusion_coeff * dt))
        frame[:, :, 1:3] += (avg[:, :, 1:3] - frame[:, :, 1:3]) * (1 - torch.exp(-viscosity * dt))

    return frame


def pressure_solve_step(frame, iterations=40):
    """
    Iteratively solves for pressure using the Poisson equation.
    """

    for _ in range(int(int(iterations))):
        divergence = frame[:, :, 3]
        pressure = frame[:, :, 4]

        up = torch.roll(pressure, -1, dims=0)
        down = torch.roll(pressure, 1, dims=0)
        left = torch.roll(pressure, -1, dims=1)
        right = torch.roll(pressure, 1, dims=1)

        frame[:, :, 4] = (up + down + left + right - divergence) / 4

    return frame


def correction_step(frame):

    pressure = frame[:, :, 4]

    up = torch.roll(pressure, -1, dims=0)
    down = torch.roll(pressure, 1, dims=0)
    left = torch.roll(pressure, -1, dims=1)
    right = torch.roll(pressure, 1, dims=1)

    frame[:, :, 1] -= 0.5 * (right - left)
    frame[:, :, 2] -= 0.5 * (up - down)

    return frame
