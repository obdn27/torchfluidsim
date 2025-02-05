import matplotlib.pyplot as plt
import numpy as np
import time
from multiprocessing import shared_memory
import torch
from config import *
from collections import deque
import frame_analysis

print("GRAPHER STARTED", __name__)

def graphing_thread():
    """
    Reads simulation data from shared memory and updates multiple time-series graphs in real-time.
    """
    shm = shared_memory.SharedMemory(name=FIELDS_BUFFER_NAME)
    buffer = np.ndarray((*GRID_RESOLUTION, 6), dtype=np.float32, buffer=shm.buf)  # Now supports all frame data

    plt.ion()
    fig, axs = plt.subplots(3, 2, figsize=(10, 10))  # 3x2 grid of plots

    metric_names = [
        "total mass", "total momentum", "kinetic energy",
        "vorticity sum", "average pressure", "drag"
    ]

    data_buffers = {name: deque(maxlen=MAX_DATA_LEN) for name in metric_names}
    x_data = deque(maxlen=MAX_DATA_LEN)

    lines = {}
    for ax, name in zip(axs.flat, metric_names):
        lines[name], = ax.plot([], [], label=name)
        ax.set_title(name.replace("_", " ").title())
        ax.set_xlabel("Time (frames)")
        ax.set_ylabel(name)
        ax.legend()

    frame_count = 0

    try:
        while True:
            # Extract frame metrics
            frame_tensor = torch.from_numpy(buffer.copy())  # Convert to tensor for analysis
            metrics = frame_analysis.analyse_frame(frame_tensor)

            x_data.append(frame_count)

            # Update each metric plot
            for name in metric_names:
                data_buffers[name].append(metrics[name])
                lines[name].set_xdata(list(x_data))
                lines[name].set_ydata(list(data_buffers[name]))

            for ax in axs.flat:
                ax.relim()
                ax.autoscale_view()

            plt.draw()
            plt.pause(1 / FPS)  # Real-time update

            frame_count += 1

    except KeyboardInterrupt:
        print("Graphing thread stopped.")
    finally:
        plt.ioff()
        plt.close()
        shm.close()

if __name__ == "__main__":
    graphing_thread()
