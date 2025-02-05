import numpy as np
import torch

def analyse_frame(frame):

    frame = frame[3:]

    density, u, v, pressure, obstacle_mask = frame[..., 0], frame[..., 1], frame[..., 2], frame[..., 4], frame[..., 5]

    total_mass = torch.sum(density)

    total_momentum = torch.sum(density.unsqueeze(-1).expand(-1, -1, 2) * torch.stack([u, v], dim=-1), dim=(0, 1, 2))

    kinetic_energy = 0.5 * torch.sum(density * (u**2 + v**2))

    du_dy = torch.roll(u, shifts=-1, dims=0) - u
    dv_dx = torch.roll(v, shifts=-1, dims=1) - v

    vorticity = dv_dx - du_dy

    circulation = torch.sum(vorticity)

    avg_pressure = torch.mean(pressure)

    drag, lift = 0, 0

    drag = torch.sum(pressure * obstacle_mask)
    lift = torch.sum(pressure * obstacle_mask)

    return {
        "total mass": total_mass.item(),
        "total momentum": total_momentum.item(),
        "kinetic energy": kinetic_energy.item(),
        "vorticity sum": circulation.item(),
        "average pressure": avg_pressure.item(),
        "drag": drag,
        "lift": lift,
    }
