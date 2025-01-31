import tkinter as tk
from tkinter import ttk
from main import update_simulation_param, create_shm_params
from config import *

def control_panel():
    """
    Tkinter control panel for adjusting simulation parameters dynamically.
    """

    def update_param(param_name):
        """
        Returns a function that updates the shared memory with the new parameter value.
        """
        return lambda val: update_simulation_param(param_name, float(val), create_shm_params())

    root = tk.Tk()
    root.geometry("300x1020+50+20")
    root.title("Simulation Parameters")

    sliders = {}  # Dictionary to store slider references

    for param_name, (default_value, min_, max_) in SIM_PARAMS_DEFAULTS.items():

        if param_name in ["mouse_x", "mouse_y", "density_scaling", "dx", "dy", "reset_request"]:
            continue

        ttk.Label(root, text=param_name.replace("_", " ").title()).pack(pady=5)
        slider = tk.Scale(
            root,
            from_=min_,
            to=max_,
            orient="horizontal",
            command=update_param(param_name),
            resolution=0.05
        )
        slider.set(default_value)
        slider.pack(pady=5)
        sliders[param_name] = slider  # Store the slider reference

    root.mainloop()
