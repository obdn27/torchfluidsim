import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

class Solver:
    def __init__(self, frame):
        """
        Initializes the solver with a given frame.
        The frame is used to set the simulation dimensions and grid mesh.
        """
        self.update_grid_size(frame)

    def update_grid_size(self, frame):
        """
        Updates the simulation grid dimensions and reinitializes the grid mesh.
        This should be called whenever the simulation grid size changes.
        """
        self.frame_device = frame.device
        self.H, self.W = frame.shape[:2]
        self.init_grid_mesh()
        self.prev_mouse_x, self.prev_mouse_y = 0, 0

    def init_grid_mesh(self):
        """Initializes the coordinate grid (X, Y) based on the current simulation dimensions."""
        self.Y, self.X = torch.meshgrid(
            torch.arange(self.H, device=self.frame_device),
            torch.arange(self.W, device=self.frame_device),
            indexing="ij",
        )

    def interaction_step(self, frame, interaction_radius, interaction_strength,
                         injection_strength, reset_request, mouse_x, mouse_y,
                         grid_resolution, window_res, dt):
        """
        Applies interaction based on mouse events, injecting velocity in the direction
        of the mouse movement (calculated from the change in mouse position) with vector arithmetic.
        """

        # Compute distance from the mouse position for every grid point.
        distance = torch.sqrt((self.X - mouse_x) ** 2 + (self.Y - mouse_y) ** 2)
        mask = distance < interaction_radius
        falloff = (1.0 - (distance / interaction_radius)).clamp(min=0) * mask.float()

        # Unpack the fields:
        # density: channel 0, velocity: channels 1-2, divergence: channel 3, pressure: channel 4, obstacle: channel 5.
        density    = frame[:, :, 0]
        vel        = frame[:, :, 1:3]  # combine x_vel and y_vel into a 2D vector field
        divergence = frame[:, :, 3]
        pressure   = frame[:, :, 4]
        obstacle   = frame[:, :, 5]

        # Calculate the mouse acceleration vector based on the change in mouse coordinates.
        # This gives the vector from the previous mouse position to the current one.
        mouse_accel = torch.tensor([mouse_x - self.prev_mouse_x, mouse_y - self.prev_mouse_y],
                                   device=frame.device, dtype=frame.dtype)
        norm = torch.sqrt((mouse_accel ** 2).sum()) + 1e-8  # avoid division by zero
        direction = mouse_accel / norm

        if injection_strength <= 1 or True:
            # Use broadcasting to apply the falloff mask to the 2D vector field.
            vel = vel + (direction * norm * falloff.unsqueeze(-1)) * interaction_strength

            # Optionally, inject some density as well.
            density = density + (100.0 * falloff * dt) * interaction_strength

        # Update previous mouse coordinates for the next interaction step.
        self.prev_mouse_x, self.prev_mouse_y = mouse_x, mouse_y

        # Reassemble the updated frame.
        updated_frame = torch.stack([
            density,
            vel[:, :, 0],
            vel[:, :, 1],
            divergence,
            pressure,
            obstacle
        ], dim=-1)
        return updated_frame.to(frame.device)


    def add_streamlines(self, frame, stream_speed, stream_spacing, stream_thickness):
        """Adds streamlines to the frame."""
        stream_thickness = min(1, int(stream_thickness))
        stream_spacing = int(stream_spacing)
        stream_speed = float(stream_speed)

        streamline_mask = ((self.X % stream_spacing) < stream_thickness).float()
        left_region_mask = (self.Y < stream_thickness).float()
        final_mask = streamline_mask * left_region_mask

        value = torch.max(frame[:, :, 0])
        value = torch.clamp(value, min=5.0)

        frame[:stream_thickness, :, 0] = ((value / 2) * final_mask)[:stream_thickness, ...]
        frame[:stream_thickness, :, 2] = (stream_speed * left_region_mask)[:stream_thickness, ...]
        return frame

    def advection_step(self, frame, dt, grid_resolution):
        """Performs a simple advection step on the frame."""
        height, width = grid_resolution
        density = frame[:, :, 0]
        x_vel = frame[:, :, 1]
        y_vel = frame[:, :, 2]
        divergence = frame[:, :, 3]
        pressure = frame[:, :, 4]
        obstacle = frame[:, :, 5]

        backtracked_x = self.X - dt * x_vel
        backtracked_y = self.Y - dt * y_vel
        backtracked_x = torch.clamp(backtracked_x / (width - 1) * 2 - 1, -1, 1)
        backtracked_y = torch.clamp(backtracked_y / (height - 1) * 2 - 1, -1, 1)

        grid = torch.stack([backtracked_x, backtracked_y], dim=-1).unsqueeze(0)

        advected_density = F.grid_sample(density.unsqueeze(0).unsqueeze(0), grid,
                                         mode='bilinear', align_corners=True).squeeze()
        advected_x_vel = F.grid_sample(x_vel.unsqueeze(0).unsqueeze(0), grid,
                                       mode='bilinear', align_corners=True).squeeze()
        advected_y_vel = F.grid_sample(y_vel.unsqueeze(0).unsqueeze(0), grid,
                                       mode='bilinear', align_corners=True).squeeze()

        advected_frame = torch.stack([advected_density, advected_x_vel,
                                      advected_y_vel, divergence, pressure, obstacle], dim=-1)
        return advected_frame

    def diffuse_step(self, frame, viscosity, diffusion_coeff, decay_rate, dt, iterations=20):
        """
        Applies diffusion (smoothing) and decay to the simulation.
        Diffusion is applied to the density (channel 0) with diffusion_coeff,
        and viscosity is applied to the velocity fields (channels 1 and 2) with viscosity.
        Additionally, decay is applied to density (channel 0) and pressure (channel 4).
        """
        # Apply decay to density and pressure
        frame[..., 0] *= decay_rate
        frame[..., 4] *= decay_rate

        # Extract fields
        density = frame[..., 0]
        x_vel = frame[..., 1]
        y_vel = frame[..., 2]

        # Compute coefficients for the Jacobi iterations.
        a_diff = diffusion_coeff * dt
        a_visc = viscosity * dt
        r_diff = 1 + 4 * a_diff
        r_visc = 1 + 4 * a_visc

        # Diffuse density using Jacobi iteration.
        for _ in range(iterations):
            density = (density +
                    a_diff * (torch.roll(density, shifts=1, dims=0) +
                                torch.roll(density, shifts=-1, dims=0) +
                                torch.roll(density, shifts=1, dims=1) +
                                torch.roll(density, shifts=-1, dims=1))) / r_diff

        # Diffuse velocity (apply viscosity) using Jacobi iteration.
        for _ in range(iterations):
            x_vel = (x_vel +
                    a_visc * (torch.roll(x_vel, shifts=1, dims=0) +
                            torch.roll(x_vel, shifts=-1, dims=0) +
                            torch.roll(x_vel, shifts=1, dims=1) +
                            torch.roll(x_vel, shifts=-1, dims=1))) / r_visc

        for _ in range(iterations):
            y_vel = (y_vel +
                    a_visc * (torch.roll(y_vel, shifts=1, dims=0) +
                            torch.roll(y_vel, shifts=-1, dims=0) +
                            torch.roll(y_vel, shifts=1, dims=1) +
                            torch.roll(y_vel, shifts=-1, dims=1))) / r_visc

        # Write the updated fields back into the frame.
        frame[..., 0] = density
        frame[..., 1] = x_vel
        frame[..., 2] = y_vel

        return frame

    @staticmethod
    def iterate_pressure(pressure, divergence, iterations, over_relaxation):
        """Iteratively solves for the pressure field."""
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

    def hierarchical_projection_step(self, frame, iterations, over_relaxation, scale_factor=4):
        """Performs a hierarchical projection step to enforce divergence-free velocity."""
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
        coarse_pressure = self.iterate_pressure(coarse_pressure, div_pyramids[-1], iterations, over_relaxation)
        pressure_pyramids.append(coarse_pressure)

        for i in range(levels - 1, -1, -1):
            fine_pressure = F.interpolate(pressure_pyramids[-1].unsqueeze(0).unsqueeze(0),
                                          size=div_pyramids[i].shape,
                                          mode='bilinear',
                                          align_corners=False).squeeze(0).squeeze(0)
            fine_pressure = self.iterate_pressure(fine_pressure, div_pyramids[i], iterations, over_relaxation)
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

    def projection_step(self, frame, iterations=40, over_relaxation=1.0, velocity_clamp=50.0):
        """Enforces a divergence-free velocity field using a pressure projection method."""
        u = frame[:, :, 1]
        v = frame[:, :, 2]
        div = (
            (torch.roll(u, shifts=-1, dims=1) - torch.roll(u, shifts=1, dims=1)) / 2 +
            (torch.roll(v, shifts=-1, dims=0) - torch.roll(v, shifts=1, dims=0)) / 2
        )
        pressure = torch.zeros_like(div, device=frame.device)
        pressure = self.iterate_pressure(pressure, div, iterations, over_relaxation)
        u = u - (torch.roll(pressure, shifts=-1, dims=1) - torch.roll(pressure, shifts=1, dims=1)) / 2
        v = v - (torch.roll(pressure, shifts=-1, dims=0) - torch.roll(pressure, shifts=1, dims=0)) / 2
        u = torch.clamp(u, -velocity_clamp, velocity_clamp)
        v = torch.clamp(v, -velocity_clamp, velocity_clamp)
        frame[:, :, 1] = u
        frame[:, :, 2] = v
        frame[:, :, 4] = pressure
        return frame
