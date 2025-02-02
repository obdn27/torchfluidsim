# config.py

SIM_PARAMS = {
    "interaction_strength": 0,
    "interaction_radius": 1,
    "decay_rate": 2,
    "simulation_speed": 3,
    "solver_iterations": 4,
    "density_scaling": 5,
    "mouse_x": 6,
    "mouse_y": 7,
    "dx": 8,
    "dy": 9,
    "reset_request": 10,
    "over_relaxation": 11,
    "injection_strength": 12,
    "stream_spacing": 13,
    "stream_thickness": 14,
    "viscosity": 15,
    "diffusion_coeff": 16,
    "current_field": 17,
}

SIM_PARAMS_DEFAULTS = {
    "interaction_strength": (48, 0.5, 64, 2),
    "interaction_radius": (12, 2, 128, 1),
    "decay_rate": (1 - 1e-2, 0.9, 1, 0.001),
    "simulation_speed": (0.1, 0.005, 0.5, 0.05),
    "solver_iterations": (50, 1, 250, 1),
    "density_scaling": (1, 0.1, 5, 0),
    "mouse_x": (0, 0, 0, 0),
    "mouse_y": (0, 0, 0, 0),
    "dx": (0, 0, 0, 0),
    "dy": (0, 0, 0, 0),
    "reset_request": (0.0, 0.0, 0.0, 0.0),
    "over_relaxation": (1.8, 0.1, 2.3, 0.1),
    "injection_strength": (45, 0, 250, 5),
    "stream_spacing": (8.0, 2.0, 32.0, 4),
    "stream_thickness": (4, 2, 8, 2),
    "viscosity": (0.01, 0, 3, 0.05),
    "diffusion_coeff": (0.01, 0, 3, 0.05),
    "current_field": (4, 0, 5, 1)
}

SIM_PARAMS_SIZE = len(SIM_PARAMS) + 1

FIELDS_BUFFER_NAME = "fields_buffer"
PARAMS_BUFFER_NAME = "params_buffer"
FILES_BUFFER_NAME = "files_buffer"

MAX_FILEPATH_SIZE = 256     # Max file path size in bytes

BASE_RES = 128
WINDOW_MULTIPLIER = 6

GRID_RESOLUTION = (BASE_RES, BASE_RES)
WINDOW_RES = (int(BASE_RES * WINDOW_MULTIPLIER), int(BASE_RES * WINDOW_MULTIPLIER))

SIM_STEPPER_LOC = "\\scripts\\sim_stepper.py"
GRAPHER_LOC = "\\scripts\\timeseries_grapher.py"

FPS = 60
MAX_DATA_LEN = 60
