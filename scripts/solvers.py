import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

H, W = 0, 0
Y, X = None, None


def init_solver(frame):

    global H, W, X, Y, image_tensor

    H, W = frame.shape[:2]

    Y, X = torch.meshgrid(
        torch.arange(H),
        torch.arange(W),
        indexing='ij',
    )


def interaction_step(frame, interaction_radius, interaction_strength, injection_strength, reset_request, mouse_x, mouse_y, grid_resolution, window_res, mouse_acceleration, dt):
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

    distance = torch.sqrt((X - norm_mouse_x) ** 2 + (Y - norm_mouse_y) ** 2)
    mask = distance < interaction_radius  # Boolean mask

    falloff = (1.0 - (distance / interaction_radius)).clamp(min=0)
    falloff = falloff * mask.float()  # Ensure zero outside radius

    density, x_vel, y_vel, divergence, pressure, obstacle = frame[:, :, 0], frame[:, :, 1], frame[:, :, 2], frame[:, :, 3], frame[:, :, 4], frame[:, :, 5]

    if injection_strength <= 1:
        x_vel = x_vel + (mouse_acceleration[0] * falloff) * interaction_strength
        y_vel = y_vel + (mouse_acceleration[1] * falloff) * interaction_strength

        density += (100.0 * falloff * dt) * interaction_strength

    updated_frame = torch.stack([density, x_vel, y_vel, divergence, pressure, obstacle], dim=-1)

    updated_frame[..., :5] *= reset_request

    return updated_frame


def add_streamlines(frame, stream_speed, stream_spacing, stream_thickness):

    stream_thickness =  torch.clamp(stream_thickness.int(), min=1).item()
    stream_spacing = stream_spacing.int().item()
    stream_speed = stream_speed.float()

    streamline_mask = (((Y[:stream_thickness, :] % stream_spacing) <= stream_thickness)).float()

    value = torch.max(frame[:, :, 0])
    value = torch.clamp(value, 5.0)

    # Inject velocity along the left boundary
    frame[:stream_thickness, :, 0] = value / 2 * streamline_mask
    frame[:stream_thickness, :, 2] = stream_speed

    return frame


def advection_step(frame, dt, grid_resolution):
    height, width = grid_resolution

    density, x_vel, y_vel, divergence, pressure, obstacle = frame[:, :, 0], frame[:, :, 1], frame[:, :, 2], frame[:, :, 3], frame[:, :, 4], frame[:, :, 5]

    # Compute backtracked positions
    backtracked_x = X - dt * x_vel
    backtracked_y = Y - dt * y_vel
    backtracked_x = torch.clamp(backtracked_x / (width - 1) * 2 - 1, -1, 1)
    backtracked_y = torch.clamp(backtracked_y / (height - 1) * 2 - 1, -1, 1)

    # Stack grid positions into a format suitable for grid_sample
    grid = torch.stack([backtracked_x, backtracked_y], dim=-1).unsqueeze(0)  # (1, H, W, 2)

    # Perform interpolation
    advected_density = F.grid_sample(density.unsqueeze(0).unsqueeze(0), grid, mode='bilinear', align_corners=True).squeeze()
    advected_x_vel = F.grid_sample(x_vel.unsqueeze(0).unsqueeze(0), grid, mode='bilinear', align_corners=True).squeeze()
    advected_y_vel = F.grid_sample(y_vel.unsqueeze(0).unsqueeze(0), grid, mode='bilinear', align_corners=True).squeeze()

    advected_frame = torch.stack([advected_density, advected_x_vel, advected_y_vel, divergence, pressure, obstacle], dim=-1)

    return advected_frame


def diffuse_step(frame, viscosity, diffusion_coeff, decay_rate, dt, iterations=20):
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

    frame[..., 0] *= decay_rate

    return frame


def iterate_pressure(pressure, divergence, iterations, over_relaxation):

    for _ in range(int(int(iterations))):  # More iterations for better convergence
        pressure = 0.25 * (
            torch.roll(pressure, shifts=1, dims=0) +
            torch.roll(pressure, shifts=-1, dims=0) +
            torch.roll(pressure, shifts=1, dims=1) +
            torch.roll(pressure, shifts=-1, dims=1) -
            divergence
        )
    
    # Apply over-relaxation
    pressure *= over_relaxation

    return pressure


def hierarchical_projection_step(frame, iterations, over_relaxation, scale_factor=4):
    """
    Multi-level pressure solve for better pressure propagation using hierarchical grids.

    Args:
    - frame (torch.Tensor): The simulation state tensor of shape (H, W, 6).
    - iterations (int): Number of iterations for solving pressure.
    - over_relaxation (float): Factor to speed up convergence.
    - scale_factor (int): Factor by which each level downscales (e.g., 2 = half, 3 = third, etc.).

    Returns:
    - frame (torch.Tensor): Updated frame with corrected velocity field.
    """

    H, W, _ = frame.shape

    # Extract velocity components and obstacle mask
    u = frame[:, :, 1]  # x-velocity
    v = frame[:, :, 2]  # y-velocity
    obstacle = frame[..., 5]  # 1 - air, 0 - obstacle

    # Compute divergence while considering obstacles
    fine_div = (
        (torch.roll(u, shifts=-1, dims=1) * torch.roll(obstacle, shifts=-1, dims=1) - 
         torch.roll(u, shifts=1, dims=1) * torch.roll(obstacle, shifts=1, dims=1)) / 2 +
        (torch.roll(v, shifts=-1, dims=0) * torch.roll(obstacle, shifts=-1, dims=0) - 
         torch.roll(v, shifts=1, dims=0) * torch.roll(obstacle, shifts=1, dims=0)) / 2
    )

    # Compute number of levels based on the scale factor
    levels = 0
    min_dim = min(H, W)
    while min_dim >= 4 * scale_factor:  # Stop when the grid is smaller than 4x4
        min_dim //= scale_factor
        levels += 1

    levels = max(1, levels)  # Ensure at least one level

    # Store pressure at different levels
    pressure_pyramids = []
    div_pyramids = [fine_div]

    # Build the hierarchy: Downsample divergence field at each level
    for _ in range(levels):
        downsampled_div = F.avg_pool2d(
            div_pyramids[-1].unsqueeze(0).unsqueeze(0), 
            kernel_size=scale_factor, stride=scale_factor
        ).squeeze(0).squeeze(0)
        div_pyramids.append(downsampled_div)

    # Solve for pressure at the coarsest level first
    coarse_pressure = torch.zeros_like(div_pyramids[-1], device=frame.device)
    coarse_pressure = iterate_pressure(coarse_pressure, div_pyramids[-1], iterations, over_relaxation)
    pressure_pyramids.append(coarse_pressure)

    # Upsample and solve progressively at each finer level
    for i in range(levels - 1, -1, -1):
        fine_pressure = F.interpolate(
            pressure_pyramids[-1].unsqueeze(0).unsqueeze(0), 
            size=div_pyramids[i].shape, mode='bilinear', align_corners=False
        ).squeeze(0).squeeze(0)

        # Solve pressure again at this level
        fine_pressure = iterate_pressure(fine_pressure, div_pyramids[i], iterations, over_relaxation)
        pressure_pyramids.append(fine_pressure)

    # Final pressure field from the finest level
    pressure = pressure_pyramids[-1]

    # Compute pressure gradients
    grad_pressure_x = (torch.roll(pressure, shifts=-1, dims=1) - torch.roll(pressure, shifts=1, dims=1)) / 2
    grad_pressure_y = (torch.roll(pressure, shifts=-1, dims=0) - torch.roll(pressure, shifts=1, dims=0)) / 2

    # Apply pressure gradient correction to velocity (collocated grid)
    u -= grad_pressure_x
    v -= grad_pressure_y

    u *= obstacle
    v *= obstacle

    # Store the updated values
    frame[..., 1] = u
    frame[..., 2] = v
    frame[..., 4] = pressure  # Store pressure for reference

    return frame


def projection_step(frame, iterations=40, over_relaxation=1.0, velocity_clamp=50.0):
    """
    Enforces incompressibility using a collocated grid (pressure & velocity at the same location).

    Args:
    - frame (torch.Tensor): The simulation state tensor of shape (H, W, 5).
    - iterations (int): Number of iterations for solving pressure.
    - over_relaxation (float): Factor to speed up convergence.
    - velocity_clamp (float): Maximum allowable velocity magnitude.

    Returns:
    - frame (torch.Tensor): Updated frame with corrected velocity field.
    """

    # Extract velocity components (stored at the same cell centers)
    u = frame[:, :, 1]  # x-velocity
    v = frame[:, :, 2]  # y-velocity

    # Compute divergence: div(U) = d(u)/dx + d(v)/dy (collocated grid)
    div = (
        (torch.roll(u, shifts=-1, dims=1) - torch.roll(u, shifts=1, dims=1)) / 2 +
        (torch.roll(v, shifts=-1, dims=0) - torch.roll(v, shifts=1, dims=0)) / 2
    )

    # Solve for pressure using Jacobi iterations
    pressure = torch.zeros_like(div, device=frame.device)

    pressure = iterate_pressure(pressure, div, iterations, over_relaxation)

    # Apply pressure gradient correction to velocity (collocated grid)
    u -= (torch.roll(pressure, shifts=-1, dims=1) - torch.roll(pressure, shifts=1, dims=1)) / 2
    v -= (torch.roll(pressure, shifts=-1, dims=0) - torch.roll(pressure, shifts=1, dims=0)) / 2

    # Clamp velocities to prevent explosion
    u = torch.clamp(u, -velocity_clamp, velocity_clamp)
    v = torch.clamp(v, -velocity_clamp, velocity_clamp)

    # Store the updated values
    frame[:, :, 1] = u
    frame[:, :, 2] = v
    frame[:, :, 4] = pressure  # Store pressure for reference

    return frame

