from multiprocessing import shared_memory
from matplotlib import pyplot as plt
import numpy as np
import threading
import subprocess
import sys
import os
import pygame
from PIL import Image

from config import *


# METHODS


def destroy_shared_memory(shm):
    shm.close()
    shm.unlink()


def create_shm(BUFFER_NAME, nbytes):
    """
    :param BUFFER_NAME: name of the buffer to locate/store data
    :param nbytes: required size of the buffer to store data
    :return: SharedMemory object
    General method to create shared memory for storing data that can be used by multiple processes.
    """

    print(f"Creating shm with BUFFER_NAME: {BUFFER_NAME} and nbytes {nbytes}")

    try:
        shm = shared_memory.SharedMemory(name=BUFFER_NAME, create=True, size=nbytes)
    except FileExistsError:
        print(f"Shared memory '{BUFFER_NAME}' exists. Cleaning up...")
        shm = shared_memory.SharedMemory(name=BUFFER_NAME)       # Locates existing shared memory by specifying create=False
        destroy_shared_memory(shm)
        print(f"Unlinked existing shared memory. Creating new buffer '{BUFFER_NAME}'")
        shm = shared_memory.SharedMemory(name=BUFFER_NAME, create=True, size=nbytes)

    return shm


def update_simulation_param(param_name, value, shm_params):
    """
    Updates given simulation parameter in shared memory
    """

    param_buffer = np.ndarray((SIM_PARAMS_SIZE,), dtype=np.float32, buffer=shm_params.buf)
    param_buffer[SIM_PARAMS[param_name]] = value


def load_obstacle_texture(image_path, grid_resolution):
    """
    Loads an obstacle texture and converts it into a simulation-ready mask.

    Args:
    - image_path (str): Path to the image file.
    - grid_resolution (tuple): Simulation grid resolution (H, W).

    Returns:
    - torch.Tensor: Binary obstacle mask (1 = solid, 0 = fluid).
    """

    H, W = grid_resolution

    # Load image and convert to grayscale
    image = Image.open(image_path).convert("L")  # Convert to grayscale
    image = image.resize((W, H))  # Resize to match simulation resolution

    # Convert image to NumPy and normalize (0-255 -> 0-1)
    obstacle_mask = np.tensor(np.array(image), dtype=np.float32) / 255.0

    # Threshold the mask (1 = solid obstacle, 0 = fluid)
    obstacle_mask = (obstacle_mask > 0.5).float()

    return obstacle_mask


def visualisation_thread():
    """
    Pygame-based visualisation thread to display the shared memory field data
    """
    global shm_params

    shm = shared_memory.SharedMemory(name=FIELDS_BUFFER_NAME)
    buffer = np.ndarray((*GRID_RESOLUTION, 3), dtype=np.float32, buffer=shm.buf)
    
    pygame.init()
    screen = pygame.display.set_mode(WINDOW_RES)
    pygame.display.set_caption("Simulation visualisation")

    last_mouse_Y, last_mouse_X = 0, 0

    reset = 1

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                keys = pygame.key.get_pressed()
                if keys[pygame.K_SPACE]:
                    reset = 0
                if keys[pygame.K_TAB]:
                    pass


        mouse_Y, mouse_X = pygame.mouse.get_pos()
        dy, dx = mouse_Y - last_mouse_Y, mouse_X - last_mouse_X

        update_simulation_param("mouse_x", mouse_X, shm_params)
        update_simulation_param("mouse_y", mouse_Y, shm_params)
        update_simulation_param("dx", dx, shm_params)
        update_simulation_param("dy", dy, shm_params)
        update_simulation_param("reset_request", reset, shm_params)

        reset = 1

        data = buffer.copy()

        # density, obstacle = data[..., 0], data[..., 1]

        # density = colormap(data)[..., :-1]
        # obstacle = colormap(obstacle)[..., :-1]

        dens_surface = pygame.surfarray.make_surface((data * 255).astype(np.uint8))
        # obs_surface = pygame.surfarray.make_surface((obstacle * 255).astype(np.uint8))

        screen.blit(pygame.transform.scale(dens_surface, WINDOW_RES), (0, 0))
        # screen.blit(pygame.transform.scale(obs_surface, WINDOW_RES), (0, 0))
        pygame.display.flip()

        last_mouse_X, last_mouse_Y = mouse_X, mouse_Y

    pygame.quit()
    shm.close()


def launch_sim_stepper():
    """
    Launches the sim_stepper.py script as a separate process
    """

    python_executable = sys.executable
    return subprocess.Popen([python_executable, os.getcwd() + SIM_STEPPER_LOC])


def launch_grapher():
    """
    Launches the timeseries_grapher.py script as a separate process
    """

    python_executable = sys.executable
    return subprocess.Popen([python_executable, os.getcwd() + GRAPHER_LOC])


def create_shm_params():
    global shm_params

    try:
        shm_params = shared_memory.SharedMemory(PARAMS_BUFFER_NAME)
    except FileNotFoundError:
        shm_params = create_shm(PARAMS_BUFFER_NAME, SIM_PARAMS_SIZE * 4)

    return shm_params

shm_params = None

if __name__ == "__main__":

    import params_window

    fields_shm = create_shm(FIELDS_BUFFER_NAME, int(np.prod(GRID_RESOLUTION) * 4 * 3))
    
    if not shm_params:
        shm_params = create_shm_params()

    file_path_shm = create_shm(FILES_BUFFER_NAME, MAX_FILEPATH_SIZE)

    sim_stepper_process = launch_sim_stepper()
    # grapher_process = launch_grapher()

    vis_thread = threading.Thread(target=visualisation_thread)
    vis_thread.start()

    try:
        params_window.control_panel()
    except KeyboardInterrupt:
        pass
    finally:
        vis_thread.join()
        sim_stepper_process.terminate()
        # grapher_process.terminate()
        destroy_shared_memory(fields_shm)
        destroy_shared_memory(shm_params)
        destroy_shared_memory(file_path_shm)
        print("Simulation stopped, subprocess terminated, and shared memory released.")
