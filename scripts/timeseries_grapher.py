import matplotlib.pyplot as plt
import numpy as np
import time
from multiprocessing import shared_memory
import torch
from config import *
from collections import deque

print("GRAPHER STARTED", __name__)

def graphing_thread():
    """
    Reads density values from shared memory and updates a time-series graph in real-time.
    """
    shm = shared_memory.SharedMemory(name=FIELDS_BUFFER_NAME)
    buffer = np.ndarray(GRID_RESOLUTION, dtype=np.float32, buffer=shm.buf)

    plt.ion()
    fig, ax = plt.subplots()
    x_data, y_data = deque(maxlen=MAX_DATA_LEN), deque(maxlen=MAX_DATA_LEN)
    line, = ax.plot(x_data, y_data, label="Average Density")

    ax.set_title("Time-Series Graph of Density")
    ax.set_xlabel("Time (frames)")
    ax.set_ylabel("Average Density")
    ax.legend()
    
    frame_count = 0

    try:
        while True:
            avg_density = np.mean(buffer)

            x_data.append(frame_count)
            y_data.append(avg_density)

            line.set_xdata(list(x_data))
            line.set_ydata(list(y_data))
            ax.relim()
            ax.autoscale_view()
            plt.draw()
            plt.pause(1 / FPS)

            frame_count += 1

    except KeyboardInterrupt:
        print("Graphing thread stopped.")
    finally:
        plt.ioff()
        plt.close()
        shm.close()

if __name__ == "__main__":
    graphing_thread()
