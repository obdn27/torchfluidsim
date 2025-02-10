# shm_manager.py
from multiprocessing import shared_memory
import numpy as np
from config import *
import time


class SharedMemorySegment:
    def __init__(self, name, size, shape, dtype=np.float32, default_value=None):
        self.name = name
        self.size = size
        self.shape = shape
        self.dtype = dtype
        try:
            # Try to attach to an existing segment.
            self.shm = shared_memory.SharedMemory(name=self.name, create=False)
        except FileNotFoundError:
            # If it does not exist, then create it.
            self.shm = shared_memory.SharedMemory(name=self.name, create=True, size=self.size)
            # If a default is provided, initialize the buffer.
            if default_value is not None:
                temp_arr = np.ndarray(self.shape, dtype=self.dtype, buffer=self.shm.buf)
                temp_arr[:] = default_value
        self.buffer = np.ndarray(self.shape, dtype=self.dtype, buffer=self.shm.buf)

    def read(self):
        """Returns a copy of the data in this shared memory."""
        return self.buffer.copy()

    def write(self, data):
        """Writes data into the shared memory buffer."""
        self.buffer[:] = data

    def destroy(self):
        """Closes and unlinks the shared memory segment."""
        self.shm.close()
        self.shm.unlink()


class SimulationParams(SharedMemorySegment):
    def __init__(self):
        # The parameters array is 1D, with SIM_PARAMS_SIZE elements, each a float32 (4 bytes)
        size = SIM_PARAMS_SIZE * 4
        shape = (SIM_PARAMS_SIZE,)
        defaults = np.zeros(SIM_PARAMS_SIZE, dtype=np.float32)
        for param_name, (default_value, _, _, _) in SIM_PARAMS_DEFAULTS.items():
            defaults[SIM_PARAMS[param_name]] = default_value
        super().__init__(name=PARAMS_BUFFER_NAME, size=size, shape=shape, dtype=np.float32, default_value=defaults)

    def update_param(self, param_name, value):
        """Update a single simulation parameter by name."""
        self.buffer[SIM_PARAMS[param_name]] = value


class FieldsBuffer(SharedMemorySegment):
    def __init__(self, max_res=MAX_RES):
        # The fields buffer is assumed to be a (MAX_RES, MAX_RES, 3) array of float32.
        shape = (max_res, max_res, 3)
        size = int(np.prod(shape)) * 4  # 4 bytes per float32
        default_value = 0  # Initialize with zeros.
        super().__init__(name=FIELDS_BUFFER_NAME, size=size, shape=shape, dtype=np.float32, default_value=default_value)


class FilePathBuffer(SharedMemorySegment):
    def __init__(self):
        # The file path buffer is a 1D array of MAX_FILEPATH_SIZE bytes.
        shape = (MAX_FILEPATH_SIZE,)
        size = MAX_FILEPATH_SIZE  # 1 byte per element.
        default_value = np.zeros(shape, dtype=np.uint8)
        super().__init__(name=FILES_BUFFER_NAME, size=size, shape=shape, dtype=np.uint8, default_value=default_value)
    
    def set_path(self, path_str):
        """Store a file path string in the shared memory.
        
        The string is encoded in UTF-8 and padded (or truncated) to MAX_FILEPATH_SIZE.
        """
        b = path_str.encode('utf-8')
        # Pad with null bytes if needed, or truncate.
        b = b.ljust(MAX_FILEPATH_SIZE, b'\x00')[:MAX_FILEPATH_SIZE]
        self.buffer[:] = np.frombuffer(b, dtype=np.uint8)
    
    def get_path(self):
        """Retrieves the file path string from the shared memory."""
        b = self.buffer.tobytes()
        return b.split(b'\x00')[0].decode('utf-8')


class SharedMemoryManager:
    def __init__(self, max_res=MAX_RES):
        self.fields_buffer = FieldsBuffer(max_res)
        self.sim_params = SimulationParams()
        self.file_path_buffer = FilePathBuffer()

    def destroy_all(self):
        self.fields_buffer.destroy()
        self.sim_params.destroy()
        self.file_path_buffer.destroy()

    def read_fields(self):
        """Returns a copy of the fields buffer as a NumPy array."""
        return self.fields_buffer.read()

    def read_params(self):
        """Returns the simulation parameters as a NumPy array view."""
        return self.sim_params.buffer

    def update_param(self, param_name, value):
        """Updates a simulation parameter by name."""
        self.sim_params.update_param(param_name, value)

    def set_file_path(self, path_str):
        """Writes a file path into shared memory."""
        self.file_path_buffer.set_path(path_str)

    def get_file_path(self):
        """Retrieves the file path from shared memory."""
        return self.file_path_buffer.get_path()


def generate_periodic_image(t):
    """Generates a 256x256x3 periodic RGB image influenced by time t."""
    x = np.linspace(0, 2 * np.pi, 256)
    y = np.linspace(0, 2 * np.pi, 256)
    X, Y = np.meshgrid(x, y)

    R = (np.sin(X + t) + 1) / 2  # Sinusoidal red channel
    G = (np.cos(Y - t) + 1) / 2   # Cosine green channel
    B = (np.sin(X + Y + t) + 1) / 2  # Combined wave blue channel

    image = np.stack([R, G, B], axis=-1)
    return image
