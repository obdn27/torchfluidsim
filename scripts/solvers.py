import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Global variables for grid dimensions and coordinate tensors.
H, W = 0, 0
Y, X = None, None


def init_solver(frame):
    global H, W, X, Y
    H, W = frame.shape[:2]
    device = frame.device
    Y, X = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing="ij",
    )


def interaction_step(frame, interaction_radius, interaction_strength, injection_strength, reset_request,
                     mouse_x, mouse_y, grid_resolution, window_res, mouse_acceleration, dt):
    norm_mouse_x = (mouse_x / window_res[0]) * grid_resolution[0]
    norm_mouse_y = (mouse_y / window_res[1]) * grid_resolution[1]

    distance = torch.sqrt((X - norm_mouse_x) ** 2 + (Y - norm_mouse_y) ** 2)
    mask = distance < interaction_radius
    falloff = (1.0 - (distance / interaction_radius)).clamp(min=0) * mask.float()

    density = frame[:, :, 0]
    x_vel = frame[:, :, 1]
    y_vel = frame[:, :, 2]
    divergence = frame[:, :, 3]
    pressure = frame[:, :, 4]
    obstacle = frame[:, :, 5]

    if injection_strength <= 1:
        x_vel = x_vel + (mouse_acceleration[0] * falloff) * interaction_strength
        y_vel = y_vel + (mouse_acceleration[1] * falloff) * interaction_strength
        density = density + (100.0 * falloff * dt) * interaction_strength

    updated_frame = torch.stack([density, x_vel, y_vel, divergence, pressure, obstacle], dim=-1).to(frame.device)

    return updated_frame


def add_streamlines(frame, stream_speed, stream_spacing, stream_thickness):
    stream_thickness = min(1, int(stream_thickness))
    stream_spacing = int(stream_spacing)
    stream_speed = float(stream_speed)

    streamline_mask = ((X % stream_spacing) < stream_thickness).float()
    left_region_mask = (Y < stream_thickness).float()
    final_mask = streamline_mask * left_region_mask

    value = torch.max(frame[:, :, 0])
    value = torch.clamp(value, min=5.0)

    frame[:stream_thickness, :, 0] = ((value / 2) * final_mask)[:stream_thickness, ...]
    frame[:stream_thickness, :, 2] = (stream_speed * left_region_mask)[:stream_thickness, ...]

    return frame


def advection_step(frame, dt, grid_resolution):
    height, width = grid_resolution
    density = frame[:, :, 0]
    x_vel = frame[:, :, 1]
    y_vel = frame[:, :, 2]
    divergence = frame[:, :, 3]
    pressure = frame[:, :, 4]
    obstacle = frame[:, :, 5]

    backtracked_x = X - dt * x_vel
    backtracked_y = Y - dt * y_vel
    backtracked_x = torch.clamp(backtracked_x / (width - 1) * 2 - 1, -1, 1)
    backtracked_y = torch.clamp(backtracked_y / (height - 1) * 2 - 1, -1, 1)

    grid = torch.stack([backtracked_x, backtracked_y], dim=-1).unsqueeze(0)

    advected_density = F.grid_sample(density.unsqueeze(0).unsqueeze(0), grid, mode='bilinear', align_corners=True).squeeze()
    advected_x_vel = F.grid_sample(x_vel.unsqueeze(0).unsqueeze(0), grid, mode='bilinear', align_corners=True).squeeze()
    advected_y_vel = F.grid_sample(y_vel.unsqueeze(0).unsqueeze(0), grid, mode='bilinear', align_corners=True).squeeze()

    advected_frame = torch.stack([advected_density, advected_x_vel, advected_y_vel, divergence, pressure, obstacle], dim=-1)
    return advected_frame


def mac_cormack_advection_step(frame, dt, grid_resolution):
    dt *= 1e-2
    u = frame[:, :, 1].unsqueeze(-1)
    v = frame[:, :, 2].unsqueeze(-1)
    u = u.expand(-1, -1, 5)
    v = v.expand(-1, -1, 5)

    q = frame[..., 0:5]
    q_star = q.clone()
    q_star -= u * dt * (torch.roll(q, shifts=-1, dims=0) - q)
    q_star -= v * dt * (torch.roll(q, shifts=-1, dims=1) - q)

    q_new = q.clone()
    q_new -= 0.5 * dt * (u * (torch.roll(q_star, shifts=-1, dims=1) - q_star))
    q_new -= 0.5 * dt * (v * (torch.roll(q_star, shifts=-1, dims=1) - q_star))

    frame[..., 0:5] = q_new
    return frame


def diffuse_step(frame, viscosity, diffusion_coeff, decay_rate, dt, iterations=20):
    frame[..., 0] *= decay_rate
    frame[..., 4] *= decay_rate
    return frame


def iterate_pressure(pressure, divergence, iterations, over_relaxation):
    for _ in range(int(iterations)):
        pressure = 0.25 * (
            torch.roll(pressure, shifts=1, dims=0) +
            torch.roll(pressure, shifts=-1, dims=0) +
            torch.roll(pressure, shifts=1, dims=1) +
            torch.roll(pressure, shifts=-1, dims=1) -
            divergence
        )
    pressure *= over_relaxation
    return pressure


def hierarchical_projection_step(frame, iterations, over_relaxation, scale_factor=4):
    H_frame, W_frame, _ = frame.shape
    u = frame[:, :, 1]
    v = frame[:, :, 2]
    obstacle = frame[..., 5]

    fine_div = (
        (torch.roll(u, shifts=-1, dims=1) * torch.roll(obstacle, shifts=-1, dims=1) -
         torch.roll(u, shifts=1, dims=1) * torch.roll(obstacle, shifts=1, dims=1)) / 2 +
        (torch.roll(v, shifts=-1, dims=0) * torch.roll(obstacle, shifts=-1, dims=0) -
         torch.roll(v, shifts=1, dims=0) * torch.roll(obstacle, shifts=1, dims=0)) / 2
    )

    levels = 0
    min_dim = min(H_frame, W_frame)
    while min_dim >= 4 * scale_factor:
        min_dim //= scale_factor
        levels += 1
    levels = max(1, levels)

    pressure_pyramids = []
    div_pyramids = [fine_div]

    for _ in range(levels):
        downsampled_div = F.avg_pool2d(div_pyramids[-1].unsqueeze(0).unsqueeze(0),
                                       kernel_size=scale_factor,
                                       stride=scale_factor).squeeze(0).squeeze(0)
        div_pyramids.append(downsampled_div)

    coarse_pressure = torch.zeros_like(div_pyramids[-1], device=frame.device)
    coarse_pressure = iterate_pressure(coarse_pressure, div_pyramids[-1], iterations, over_relaxation)
    pressure_pyramids.append(coarse_pressure)

    for i in range(levels - 1, -1, -1):
        fine_pressure = F.interpolate(pressure_pyramids[-1].unsqueeze(0).unsqueeze(0),
                                      size=div_pyramids[i].shape,
                                      mode='bilinear',
                                      align_corners=False).squeeze(0).squeeze(0)
        fine_pressure = iterate_pressure(fine_pressure, div_pyramids[i], iterations, over_relaxation)
        pressure_pyramids.append(fine_pressure)

    pressure = pressure_pyramids[-1]
    grad_pressure_x = (torch.roll(pressure, shifts=-1, dims=1) - torch.roll(pressure, shifts=1, dims=1)) / 2
    grad_pressure_y = (torch.roll(pressure, shifts=-1, dims=0) - torch.roll(pressure, shifts=1, dims=0)) / 2

    u = u - grad_pressure_x
    v = v - grad_pressure_y
    u = u * obstacle
    v = v * obstacle

    frame[..., 1] = u
    frame[..., 2] = v
    frame[..., 4] = pressure

    return frame


def projection_step(frame, iterations=40, over_relaxation=1.0, velocity_clamp=50.0):
    u = frame[:, :, 1]
    v = frame[:, :, 2]
    div = (
        (torch.roll(u, shifts=-1, dims=1) - torch.roll(u, shifts=1, dims=1)) / 2 +
        (torch.roll(v, shifts=-1, dims=0) - torch.roll(v, shifts=1, dims=0)) / 2
    )
    pressure = torch.zeros_like(div, device=frame.device)
    pressure = iterate_pressure(pressure, div, iterations, over_relaxation)
    u = u - (torch.roll(pressure, shifts=-1, dims=1) - torch.roll(pressure, shifts=1, dims=1)) / 2
    v = v - (torch.roll(pressure, shifts=-1, dims=0) - torch.roll(pressure, shifts=1, dims=0)) / 2
    u = torch.clamp(u, -velocity_clamp, velocity_clamp)
    v = torch.clamp(v, -velocity_clamp, velocity_clamp)
    frame[:, :, 1] = u
    frame[:, :, 2] = v
    frame[:, :, 4] = pressure
    return frame
