from multiprocessing import shared_memory
import time
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from config import *
import solvers
from shm_manager import SharedMemoryManager
import shm_manager

colormap = plt.cm.inferno


class SimulationStepper:
    def __init__(self, shm_manager=None, base_res=BASE_RES, max_res=MAX_RES, fps=FPS):
        self.base_res = base_res
        self.max_res = max_res
        self.fps = fps
        self.device = "cpu"  # Change as needed (e.g., "cuda" or "mps")
        self.grid_resolution = (base_res, base_res)

        # Use the provided shared memory manager or create a new one.
        self.shm_manager = shm_manager if shm_manager is not None else SharedMemoryManager()
        self._init_shared_memory()

        self.current_frame = torch.zeros(
            (base_res, base_res, 6),
            dtype=torch.float32,
            device=self.device,
        )
        self.image_tensor = torch.zeros(
            (base_res, base_res, 3),
            device=self.device,
        )
        self.obstacle_path = ""
        self.max_res_frame = torch.zeros((max_res, max_res, 3), device=self.device)
        solvers.init_solver(self.current_frame)

    def _init_shared_memory(self):
        # Use the shared memory manager’s buffers.
        self.vis_buffer = torch.from_numpy(self.shm_manager.fields_buffer.buffer).to(self.device)
        self.params_buffer = self.shm_manager.sim_params.buffer  # NumPy array view (updates immediately)
        self.filepath_buffer = self.shm_manager.file_path_buffer.buffer  # NumPy array (dtype=uint8)

    def load_obstacle_texture(self, image_path):
        H, W = self.grid_resolution
        img = Image.open(image_path).convert("L").resize((W, H))
        arr = np.array(img)
        mask = torch.tensor(arr, dtype=torch.float32, device=self.device) / 255.0
        mask = (mask < 0.5).float()
        texture = torch.tensor(arr, device=self.device)
        # Return a binary obstacle mask and a 3-channel image tensor.
        return 1 - mask.permute(1, 0), texture.permute(1, 0).unsqueeze(-1).expand(-1, -1, 3)

    def update_obstacle_texture(self, new_path):
        if new_path and new_path != self.obstacle_path:
            self.current_frame.zero_()
            self.current_frame[..., 5], self.image_tensor = self.load_obstacle_texture(new_path)
            self.obstacle_path = new_path

    @staticmethod
    def normalize_array(arr):
        mn, mx = torch.min(arr), torch.max(arr)
        return (arr - mn) / (mx - mn) if mx > mn else torch.zeros_like(arr)

    def process_frame(self, frame, field):
        # field may come in as a scalar (or 0-dim tensor); convert to int.
        field = int(field)
        norm = self.normalize_array(frame[..., field] * frame[..., 5])
        overlay = self.image_tensor * (1 - frame[..., 5]).unsqueeze(-1).expand_as(self.image_tensor)
        colored = torch.from_numpy(colormap(norm.cpu())[..., :3])

        return colored + overlay.cpu()

    def pad_frame(self, frame, target_size):
        target_size = int(target_size)
        if frame.shape[0] != target_size or frame.shape[1] != target_size:
            frame = F.interpolate(
                frame.permute(2, 0, 1).unsqueeze(0),
                size=(target_size, target_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).permute(1, 2, 0)
        self.max_res_frame[:target_size, :target_size, :] = frame
        return self.max_res_frame

    def simulation_step(self):
        self.current_frame = step_simulation(self.current_frame, self.params_buffer, self.grid_resolution)

    def update_grid_resolution(self):
        # self.params_buffer is a NumPy array, so use int(...) directly.
        new_width = int(self.params_buffer[SIM_PARAMS["grid_width"]])
        if new_width != self.grid_resolution[0]:
            self.grid_resolution = (new_width, new_width)
            self.current_frame = torch.zeros(
                (new_width, new_width, 6), dtype=torch.float32, device=self.device
            )
            self.image_tensor = torch.zeros((new_width, new_width, 3), device=self.device)
            solvers.init_solver(self.current_frame)

    def run(self):
        while True:
            # Read the file path from the shared memory file path buffer.
            raw_path = bytes(self.filepath_buffer[:MAX_FILEPATH_SIZE]).decode("utf-8")
            new_path = raw_path.strip().strip("\x00")
            self.update_obstacle_texture(new_path)
            self.update_grid_resolution()
            # print("current field", self.params_buffer[SIM_PARAMS["current_field"]])
            # Uncomment the following lines to perform simulation steps and update the visual buffer:
            self.simulation_step()
            proc = self.process_frame(self.current_frame, self.params_buffer[SIM_PARAMS["current_field"]])
            padded = self.pad_frame(proc, self.params_buffer[SIM_PARAMS["grid_width"]])
            self.vis_buffer.copy_(padded)
            time.sleep(1 / self.fps)


def step_simulation(current_frame, params, grid_resolution):
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
    frame = solvers.add_streamlines(
        frame=frame,
        stream_speed=params[SIM_PARAMS["injection_strength"]],
        stream_spacing=params[SIM_PARAMS["stream_spacing"]],
        stream_thickness=params[SIM_PARAMS["stream_thickness"]],
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
    frame = solvers.hierarchical_projection_step(
        frame=frame,
        iterations=params[SIM_PARAMS["solver_iterations"]],
        over_relaxation=params[SIM_PARAMS["over_relaxation"]],
        scale_factor=2,
    )
    return frame
