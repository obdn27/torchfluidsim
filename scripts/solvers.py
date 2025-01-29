import torch
import numpy as np

def interaction_step(frame, interaction_radius, mouse_x, mouse_y, grid_resolution, window_res):
    norm_mouse_x = (mouse_x / window_res[0]) * grid_resolution[0]
    norm_mouse_y = (mouse_y / window_res[1]) * grid_resolution[1]

    distance = torch.sqrt((x - norm_mouse_x) ** 2 + (y - norm_mouse_y) ** 2)
    mask = distance <= interaction_radius  # Boolean mask (1 inside, 0 outside)




# def advect(frame):
