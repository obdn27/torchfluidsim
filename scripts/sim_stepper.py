from multiprocessing import shared_memory
import time
import torch
from config import *
import solvers
import numpy as np
import time
from PIL import Image
import matplotlib.pyplot as plt
import scipy.ndimage

print("SIM_STEPPER STARTED", __name__)
colormap = plt.cm.inferno
image_tensor = None

def step_simulation(current_frame, params, grid_resolution):
    """
    Steps the current simulation state forward 1 tick/frame.
    Uses PyTorch tensors instead of NumPy.
    """

    frame = solvers.interaction_step(
        frame=current_frame,
        interaction_radius=params[SIM_PARAMS["interaction_radius"]],
        interaction_strength=params[SIM_PARAMS["interaction_strength"]],
        injection_strength=params[SIM_PARAMS["injection_strength"]],
        reset_request=params[SIM_PARAMS["reset_request"]],
        mouse_x=params[SIM_PARAMS["mouse_x"]],
        mouse_y=params[SIM_PARAMS["mouse_y"]],
        grid_resolution=grid_resolution,
        window_res=WINDOW_RES,
        mouse_acceleration=(params[SIM_PARAMS["dx"]], params[SIM_PARAMS["dy"]]),
        dt=params[SIM_PARAMS["simulation_speed"]],
    )

    frame = solvers.advection_step(
        frame=frame,
        dt=params[SIM_PARAMS["simulation_speed"]],
        grid_resolution=grid_resolution,
    )

    frame = solvers.diffuse_step(
        frame=frame,
        viscosity=params[SIM_PARAMS["viscosity"]],
        diffusion_coeff=params[SIM_PARAMS["diffusion_coeff"]],
        decay_rate=params[SIM_PARAMS["decay_rate"]],
        dt=params[SIM_PARAMS["simulation_speed"]],
    )

    frame = solvers.add_streamlines(
        frame=frame,
        stream_speed=params[SIM_PARAMS["injection_strength"]],
        stream_spacing=params[SIM_PARAMS["stream_spacing"]],
        stream_thickness=params[SIM_PARAMS["stream_thickness"]],
    )
   
    frame = solvers.hierarchical_projection_step(
        frame=frame,
        iterations=params[SIM_PARAMS["solver_iterations"]],
        over_relaxation=params[SIM_PARAMS["over_relaxation"]],
        scale_factor=2,
    )

    return frame


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
    obstacle_mask = torch.tensor(np.array(image), dtype=torch.float32) / 255.0

    # Threshold the mask (0 = solid obstacle, 1 = fluid)
    obstacle_mask = (obstacle_mask < 0.5).float()

    return 1 - obstacle_mask.permute(1, 0), torch.tensor(np.array(image)).permute(1, 0).unsqueeze(-1).expand(-1, -1, 3)


def checkload_obstacle_texture(frame, old_path, new_path, grid_resolution):

    global image_tensor

    if old_path != new_path and new_path != "":
        frame[...] = 0
        frame[..., 5], image_tensor = load_obstacle_texture(new_path, grid_resolution=grid_resolution)
        print("filepath string:", new_path)

    return frame


def process_frame(frame, field):

    def normalize_array(arr):
        min_val = torch.min(arr)
        max_val = torch.max(arr)
        return (arr - min_val) / (max_val - min_val) if max_val > min_val else torch.zeros_like(arr)

    field = int(field.item())
    result = normalize_array(frame[..., field] * frame[..., 5])

    image_overlay = (image_tensor * (1 - frame[..., 5]).unsqueeze(-1).expand(-1, -1, 3))

    result = colormap(result)
    result = apply_bloom(result, 0.6)

    return torch.tensor(result)[..., :3] + image_overlay


def apply_bloom(image, threshold=0.6, blur_radius=10, intensity=0.5):
    """
    Applies a bloom effect to an HxWx4 RGBA image array.
    
    Parameters:
    - image (np.ndarray): HxWx4 image array with values in [0, 1].
    - threshold (float): Intensity threshold for bloom (0-1).
    - blur_radius (int): Radius of Gaussian blur.
    - intensity (float): How strong the bloom effect is when blended back.
    
    Returns:
    - np.ndarray: Image with bloom effect.
    """
    # Extract RGB channels (ignore alpha for bloom calculation)
    rgb = image[..., :3]

    # Convert to grayscale intensity
    grayscale = np.mean(rgb, axis=-1, keepdims=True)
    
    # Extract bright areas
    bright_areas = np.where(grayscale > threshold, rgb, np.zeros_like(rgb))
    
    # Apply Gaussian blur to each RGB channel
    blurred = np.stack([scipy.ndimage.gaussian_filter(bright_areas[..., i], blur_radius) for i in range(3)], axis=-1)
    
    # Blend with original image
    result_rgb = np.clip(rgb + intensity * blurred, 0, 1)

    # Preserve alpha channel
    result = np.concatenate([result_rgb, image[..., 3:4]], axis=-1)  # Append alpha back

    return result


def sim_stepper(grid_resolution):
    """
    Runs the simulation stepper, using PyTorch tensors for computations.
    Writes only the density field (single-channel) to shared memory for visualization.
    """

    global image_tensor

    vis_shm = shared_memory.SharedMemory(name=FIELDS_BUFFER_NAME)
    vis_buffer_np = np.ndarray((*grid_resolution, 3), dtype=np.float32, buffer=vis_shm.buf)
    vis_buffer = torch.from_numpy(vis_buffer_np).to('cuda' if torch.cuda.is_available() else 'cpu')

    params_shm = shared_memory.SharedMemory(name=PARAMS_BUFFER_NAME)
    params_np = np.ndarray((SIM_PARAMS_SIZE,), dtype=np.float32, buffer=params_shm.buf)
    params_buffer = torch.from_numpy(params_np).to('cuda' if torch.cuda.is_available() else 'cpu')

    filepath_shm = shared_memory.SharedMemory(FILES_BUFFER_NAME)
    new_path = bytes(filepath_shm.buf[:MAX_FILEPATH_SIZE]).decode('utf-8').strip()
    new_path = ''.join(c for c in new_path if ord(c) != 0)

    image_tensor = torch.zeros((*grid_resolution, 3))

    # Creates empty 6-channel frame with channels corresponding to (density, xvel, yvel, divergence, pressure, obstacle)
    current_frame = torch.zeros((*grid_resolution, 6), dtype=torch.float32, device=vis_buffer.device)

    old_path = ""

    current_frame = checkload_obstacle_texture(current_frame, old_path, new_path, grid_resolution)

    lasttime = time.time()

    solvers.init_solver(current_frame)

    while True:

        new_path = bytes(filepath_shm.buf[:MAX_FILEPATH_SIZE]).decode('utf-8').strip()
        new_path = ''.join(c for c in new_path if ord(c) != 0)

        current_frame = checkload_obstacle_texture(current_frame, old_path, new_path, grid_resolution)

        next_frame = step_simulation(current_frame, params_buffer, grid_resolution)

        vis_buffer.copy_(process_frame(next_frame, params_buffer[SIM_PARAMS["current_field"]]))  # Copy processed frame depending on which field the user has selected

        time.sleep(1 / FPS)

        old_path = new_path
        current_frame = next_frame.clone()

        currenttime = time.time()

        # print(f"{(currenttime - lasttime) * 1000:.2f}ms")
        lasttime = currenttime

if __name__ == "__main__":
    sim_stepper(GRID_RESOLUTION)
