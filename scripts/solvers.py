import torch
import numpy as np
import torch.nn.functional as F

H, W = 0, 0
Y, X = None, None


def init_solver(frame):

    global H, W, X, Y

    H, W = frame.shape[:2]

    Y, X = torch.meshgrid(
        torch.arange(H),
        torch.arange(W),
        indexing='ij',
    )


def interaction_step(frame, interaction_radius, interaction_strength, reset_request, mouse_x, mouse_y, grid_resolution, window_res, mouse_acceleration, dt, decay_rate):
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

    Y, X = torch.meshgrid(
        torch.arange(height, dtype=torch.float32, device=frame.device),
        torch.arange(width, dtype=torch.float32, device=frame.device),
        indexing='ij'
    )

    distance = torch.sqrt((X - norm_mouse_x) ** 2 + (Y - norm_mouse_y) ** 2)
    mask = distance < interaction_radius  # Boolean mask

    falloff = (1.0 - (distance / interaction_radius)).clamp(min=0)
    falloff = falloff * mask.float()  # Ensure zero outside radius

    density, x_vel, y_vel, divergence, pressure = frame[:, :, 0], frame[:, :, 1], frame[:, :, 2], frame[:, :, 3], frame[:, :, 4]

    x_vel = x_vel + (mouse_acceleration[0] * falloff)
    y_vel = y_vel + (mouse_acceleration[1] * falloff)

    density += (100.0 * falloff * dt) * interaction_strength

    updated_frame = torch.stack([density, x_vel, y_vel, divergence, pressure], dim=-1)

    return updated_frame * decay_rate * reset_request


def add_streamlines(frame, streamline_speed, streamline_spacing, streamline_thickness):

    mask = (torch.floor(Y / streamline_thickness) % int(streamline_spacing / streamline_thickness) == 0) & (X < 6)

    frame[:, :, 0] += mask
    frame[:, :, 1] += mask * streamline_speed

    return frame


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

    H, W = frame.shape[:2]

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

    for _ in range(int(int(iterations))):  # More iterations for better convergence
        pressure = 0.25 * (
            torch.roll(pressure, shifts=1, dims=0) +
            torch.roll(pressure, shifts=-1, dims=0) +
            torch.roll(pressure, shifts=1, dims=1) +
            torch.roll(pressure, shifts=-1, dims=1) -
            div
        )
    
    # Apply over-relaxation
    pressure *= over_relaxation

    # Apply pressure gradient correction to velocity (collocated grid)
    u -= (torch.roll(pressure, shifts=-1, dims=1) - torch.roll(pressure, shifts=1, dims=1)) / 2
    v -= (torch.roll(pressure, shifts=-1, dims=0) - torch.roll(pressure, shifts=1, dims=0)) / 2

    # u = torch.clamp(u, -velocity_clamp, velocity_clamp)
    # v = torch.clamp(v, -velocity_clamp, velocity_clamp)

    # Store the updated values
    frame[:, :, 1] = u
    frame[:, :, 2] = v
    frame[:, :, 4] = pressure  # Store pressure for reference

    return frame

