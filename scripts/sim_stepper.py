from multiprocessing import shared_memory
import time
import numpy as np
import torch
from config import *
import solvers

GRID_RESOLUTION = (1024, 1024)
FPS = 60

FIELDS_BUFFER_NAME = "fields_buffer"
PARAMS_BUFFER_NAME = "params_buffer"

print("SIM_STEPPER STARTED", __name__)

def step_simulation(current_frame, frame_number, grid_resolution):
    height, width = grid_resolution
    y = torch.linspace(-1, 1, steps=height).view(-1, 1).expand(height, width)
    x = torch.linspace(-1, 1, steps=width).expand(height, width)

    ripple = torch.sin(10 * torch.sqrt(x**2 + y**2) - frame_number * 0.1) * 0.5 + 0.5
    sine_wave = torch.sin(2 * torch.pi * (x + frame_number * 0.01)) * 0.5 + 0.5
    gradient = (x + y) * 0.5 + 0.5

    pattern = (ripple * 0.5 + sine_wave * 0.3 + gradient * 0.2).clamp(0, 1)

    return pattern


def read_simulation_params(params_buffer):
    """
    Reads simulation parameters from the shared memory buffer.
    """

    # for key, value in SIM_PARAMS.items():
    #     param_buffer = 

    interaction_strength = params_buffer[SIM_PARAMS["interaction_strength"]]
    simulation_speed = params_buffer[SIM_PARAMS["simulation_speed"]]

    return params_buffer


def sim_stepper(grid_resolution):
    shm = shared_memory.SharedMemory(name=FIELDS_BUFFER_NAME)
    buffer = np.ndarray(grid_resolution, dtype=np.float32, buffer=shm.buf)

    params_shm = shared_memory.SharedMemory(name=PARAMS_BUFFER_NAME)
    params_buffer = np.ndarray((SIM_PARAMS_SIZE,), dtype=np.float32, buffer=params_shm.buf)

    current_frame = np.zeros(grid_resolution)

    frame_number = 0

    while True:
        next_frame = step_simulation(current_frame, frame_number, grid_resolution)
        np.copyto(buffer, next_frame.numpy())
        params = read_simulation_params(params_buffer)

        frame_number += 10 * params[SIM_PARAMS["simulation_speed"]]
        time.sleep(1 / FPS)

        current_frame = next_frame

if __name__ == "__main__":
    sim_stepper(GRID_RESOLUTION)