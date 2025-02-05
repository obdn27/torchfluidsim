import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from main import update_simulation_param, create_shm_params, restart_simulation
from config import *
from multiprocessing import shared_memory

def control_panel():
    """
    Tkinter control panel for adjusting simulation parameters dynamically.
    """

    def update_param(param_name):
        """ Returns a function that updates the shared memory with the new parameter value. """
        return lambda val: update_simulation_param(param_name, float(val), create_shm_params())

    root = tk.Tk()
    root.geometry("450x960+16+16")
    root.title("Simulation Control Panel")

    # Create a frame to hold UI elements neatly
    container = ttk.Frame(root, padding=10)
    container.pack(fill="both", expand=True)

    # Grid Layout Configuration
    container.columnconfigure(0, weight=1)
    container.columnconfigure(1, weight=1)

    # Parameter Sliders
    param_frame = ttk.LabelFrame(container, text="Simulation Parameters", padding=10)
    param_frame.grid(row=0, column=0, columnspan=2, sticky="nsew", padx=10, pady=10)

    sliders = {}

    row_index = 0
    for param_name, (default_value, min_, max_, step) in SIM_PARAMS_DEFAULTS.items():
        if param_name in ["mouse_x", "mouse_y", "density_scaling", "dx", "dy", "reset_request", "obstacle_path", "grid_width"]:
            continue

        ttk.Label(param_frame, text=param_name.replace("_", " ").title()).grid(row=row_index, column=0, sticky="w", padx=5, pady=2)
        slider = tk.Scale(
            param_frame,
            from_=min_,
            to=max_,
            orient="horizontal",
            command=update_param(param_name),
            resolution=step,
            length=200
        )
        slider.set(default_value)
        slider.grid(row=row_index, column=1, sticky="ew", padx=5, pady=2)
        sliders[param_name] = slider
        row_index += 1

    # Grid Size Control
    grid_frame = ttk.LabelFrame(container, text="Grid Size (Square)", padding=10)
    grid_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=10, pady=10)

    def update_grid_size(val):
        """ Updates the shared memory with the new grid size. """
        size = int(val)
        update_simulation_param("grid_width", size, create_shm_params())

    grid_size_slider = tk.Scale(
        grid_frame,
        from_=64, to=2048, resolution=64,
        orient="horizontal",
        command=update_grid_size,
        length=128
    )
    grid_size_slider.set(GRID_RESOLUTION[0])  # Set default to current grid size
    grid_size_slider.pack(pady=5)

    # File Selection for Obstacle
    file_path_shm = shared_memory.SharedMemory(FILES_BUFFER_NAME)

    def get_obstacle_path():
        """ Opens a file dialog for the user to select an obstacle image and updates shared memory. """
        file_path = filedialog.askopenfilename(title="Select Obstacle Image", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
        if file_path:
            file_path_shm.buf[:MAX_FILEPATH_SIZE] = bytes(file_path.ljust(MAX_FILEPATH_SIZE), 'utf-8')

    obstacle_button = tk.Button(container, text="Load Obstacle", command=get_obstacle_path)
    obstacle_button.grid(row=3, column=0, columnspan=2, pady=10, sticky="ew")

    def quit_simulation():
        root.destroy()

    quit_button = tk.Button(container, text="Quit simulation", command=quit_simulation)
    quit_button.grid(row=4, column=0, columnspan=2, pady=10, sticky="ew")

    root.mainloop()
