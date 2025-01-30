# config.py

SIM_PARAMS = {
    "interaction_strength": 0,
    "interaction_radius": 1,
    "viscosity": 2,
    "diffusion_rate": 3,
    "decay_rate": 4,
    "simulation_speed": 5,
    "solver_iterations": 6,
    "density_scaling": 7,
    "mouse_x": 8,
    "mouse_y": 9,
    "dx": 10,
    "dy": 11,
    "vorticity_strength": 12,
}

SIM_PARAMS_DEFAULTS = {
    "interaction_strength": (1, 0.5, 64),
    "interaction_radius": (48, 2, 128),
    "viscosity": (0.1, 0, 2.5),
    "diffusion_rate": (1, 0.01, 5),
    "decay_rate": (1 - 1e-2, 0.9, 1),
    "simulation_speed": (0.3, 0.005, 2),
    "solver_iterations": (10, 1, 75),
    "density_scaling": (1, 0.1, 5),
    "mouse_x": (0, 0, 0),
    "mouse_y": (0, 0, 0),
    "dx": (0, 0, 0),
    "dy": (0, 0, 0),
    "vorticity_strength": (0.5, 0.05, 1.5),
}

SIM_PARAMS_SIZE = len(SIM_PARAMS) + 1

FIELDS_BUFFER_NAME = "fields_buffer"
PARAMS_BUFFER_NAME = "params_buffer"

BASE_RES = 384
WINDOW_MULTIPLIER = 2

GRID_RESOLUTION = (BASE_RES, BASE_RES)
WINDOW_RES = (int(BASE_RES * WINDOW_MULTIPLIER), int(BASE_RES * WINDOW_MULTIPLIER))

SIM_STEPPER_LOC = "\\scripts\\sim_stepper.py"
GRAPHER_LOC = "\\scripts\\timeseries_grapher.py"

FPS = 60
MAX_DATA_LEN = 60
