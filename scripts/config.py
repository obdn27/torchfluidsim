# config.py

SIM_PARAMS = {
    "interaction_strength": 0,
    "interaction_radius": 1,
    "viscosity": 2,
    "diffusion_rate": 3,
    "damping": 4,
    "simulation_speed": 5,
    "solver_iterations": 6,
    "density_scaling": 7,
    "mouse_x": 8,
    "mouse_y": 9,
}

SIM_PARAMS_DEFAULTS = {
    "interaction_strength": (1, 0.5, 64),
    "interaction_radius": (1, 0.5, 64),
    "viscosity": (0.1, 0, 2.5),
    "diffusion_rate": (1, 0.01, 5),
    "damping": (1 - 1e-2, 0.9, 1),
    "simulation_speed": (1, 0.05, 10),
    "solver_iterations": (10, 1, 75),
    "density_scaling": (1, 0.1, 5),
    "mouse_x": (0, 0, 0),
    "mouse_y": (0, 0, 0),
}

SIM_PARAMS_SIZE = len(SIM_PARAMS) + 1

FIELDS_BUFFER_NAME = "fields_buffer"
PARAMS_BUFFER_NAME = "params_buffer"

GRID_RESOLUTION = (1024, 1024)
WINDOW_RES = (768, 768)

SIM_STEPPER_LOC = "\\scripts\\sim_stepper.py"
