from multiprocessing import shared_memory
import time
import numpy as np
import torch
from config import *
import solvers

print("SIM_STEPPER STARTED", __name__)

def step_simulation(current_frame, frame_number, params, grid_resolution):
    """
    Generates a pattern and masks out a circular region around (mouse_x, mouse_y).
    """
    mouse_x, mouse_y = params[SIM_PARAMS["mouse_x"]], params[SIM_PARAMS["mouse_y"]]
    interaction_radius = params[SIM_PARAMS["interaction_radius"]]

    height, width = grid_resolution
    y = torch.linspace(0, height - 1, steps=height).view(-1, 1).expand(height, width)
    x = torch.linspace(0, width - 1, steps=width).expand(height, width)

    norm_mouse_x = (mouse_x / WINDOW_RES[0]) * width
    norm_mouse_y = (mouse_y / WINDOW_RES[1]) * height

    distance = torch.sqrt((x - norm_mouse_x) ** 2 + (y - norm_mouse_y) ** 2)
    mask = distance <= interaction_radius  # Boolean mask (1 inside, 0 outside)

    ripple = torch.sin(10 * torch.sqrt(x**2 + y**2) - frame_number * 0.1) * 0.5 + 0.5
    sine_wave = torch.sin(2 * torch.pi * (x + frame_number * 0.01)) * 0.5 + 0.5
    gradient = (x + y) * 0.5 + 0.5
    pattern = (ripple * 0.5 + sine_wave * 0.3 + gradient * 0.2).clamp(0, 1)

    pattern[mask] = 0.0

    return pattern


def sim_stepper(grid_resolution):
    shm = shared_memory.SharedMemory(name=FIELDS_BUFFER_NAME)
    buffer = np.ndarray(grid_resolution, dtype=np.float32, buffer=shm.buf)

    params_shm = shared_memory.SharedMemory(name=PARAMS_BUFFER_NAME)
    params_buffer = np.ndarray((SIM_PARAMS_SIZE,), dtype=np.float32, buffer=params_shm.buf)

    current_frame = np.zeros(grid_resolution)

    frame_number = 0

    while True:
        next_frame = step_simulation(current_frame, frame_number, params_buffer, grid_resolution)
        np.copyto(buffer, next_frame.numpy())

        frame_number += 10 * params_buffer[SIM_PARAMS["simulation_speed"]]
        time.sleep(1 / FPS)

        current_frame = next_frame

if __name__ == "__main__":
    sim_stepper(GRID_RESOLUTION)