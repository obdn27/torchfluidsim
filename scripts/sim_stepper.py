from multiprocessing import shared_memory
import time
import torch
from config import *
import solvers
import numpy as np
import time

print("SIM_STEPPER STARTED", __name__)

def step_simulation(current_frame, params, grid_resolution):
    """
    Steps the current simulation state forward 1 tick/frame.
    Uses PyTorch tensors instead of NumPy.
    """

    frame = solvers.interaction_step(
        frame=current_frame,
        interaction_radius=params[SIM_PARAMS["interaction_radius"]],
        interaction_strength=params[SIM_PARAMS["interaction_strength"]],
        reset_request=params[SIM_PARAMS["reset_request"]],
        mouse_x=params[SIM_PARAMS["mouse_x"]],
        mouse_y=params[SIM_PARAMS["mouse_y"]],
        grid_resolution=grid_resolution,
        window_res=WINDOW_RES,
        mouse_acceleration=(params[SIM_PARAMS["dx"]], params[SIM_PARAMS["dy"]]),
        dt=params[SIM_PARAMS["simulation_speed"]],
        decay_rate=params[SIM_PARAMS["decay_rate"]],
    )

    frame = solvers.advection_step(
        frame=frame,
        dt=params[SIM_PARAMS["simulation_speed"]],
        grid_resolution=grid_resolution,
    )

    frame = solvers.projection_step(
        frame=frame,
        iterations=params[SIM_PARAMS["solver_iterations"]],
        over_relaxation=params[SIM_PARAMS["over_relaxation"]],
    )

    return frame


def sim_stepper(grid_resolution):
    """
    Runs the simulation stepper, using PyTorch tensors for computations.
    Writes only the density field (single-channel) to shared memory for visualization.
    """

    vis_shm = shared_memory.SharedMemory(name=FIELDS_BUFFER_NAME)
    vis_buffer_np = np.ndarray(grid_resolution, dtype=np.float32, buffer=vis_shm.buf)
    vis_buffer = torch.from_numpy(vis_buffer_np).to('cuda' if torch.cuda.is_available() else 'cpu')

    params_shm = shared_memory.SharedMemory(name=PARAMS_BUFFER_NAME)
    params_np = np.ndarray((SIM_PARAMS_SIZE,), dtype=np.float32, buffer=params_shm.buf)
    params_buffer = torch.from_numpy(params_np).to('cuda' if torch.cuda.is_available() else 'cpu')

    # Creates empty 5-channel frame with channels corresponding to (density, xvel, yvel, divergence, pressure)
    current_frame = torch.zeros((*grid_resolution, 5), dtype=torch.float32, device=vis_buffer.device)

    frame_number = 0

    lasttime = time.time()

    while True:
        next_frame = step_simulation(current_frame, params_buffer, grid_resolution)

        vis_buffer.copy_(next_frame[..., 0])  # Copy only the first channel (Density)

        frame_number += 10 * params_buffer[SIM_PARAMS["simulation_speed"]]

        time.sleep(1 / FPS)

        current_frame = next_frame.clone()

        currenttime = time.time()

        print(f"{(currenttime - lasttime) * 1000:.2f}ms")
        lasttime = currenttime

if __name__ == "__main__":
    sim_stepper(GRID_RESOLUTION)
