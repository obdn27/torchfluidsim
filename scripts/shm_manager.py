from multiprocessing import shared_memory
import numpy as np
from config import *

def destroy_shared_memory(shm):
    shm.close()
    shm.unlink()


def create_shm(BUFFER_NAME, nbytes):
    print(f"Creating shm with BUFFER_NAME: {BUFFER_NAME} and nbytes {nbytes}")

    try:
        shm = shared_memory.SharedMemory(name=BUFFER_NAME, create=True, size=nbytes)
    except FileExistsError:
        print(f"Shared memory '{BUFFER_NAME}' exists. Cleaning up...")
        shm = shared_memory.SharedMemory(name=BUFFER_NAME)       # Locates existing shared memory by specifying create=False
        destroy_shared_memory(shm)
        print(f"Unlinked existing shared memory. Creating new buffer '{BUFFER_NAME}'")
        shm = shared_memory.SharedMemory(name=BUFFER_NAME, create=True, size=nbytes)

    return shm


def update_simulation_param(param_name, value, shm_params):

    param_buffer = np.ndarray((SIM_PARAMS_SIZE,), dtype=np.float32, buffer=shm_params.buf)
    param_buffer[SIM_PARAMS[param_name]] = value


def create_shm_params():

    try:
        shm_params = shared_memory.SharedMemory(PARAMS_BUFFER_NAME)
    except FileNotFoundError:
        shm_params = create_shm(PARAMS_BUFFER_NAME, SIM_PARAMS_SIZE * 4)

    return shm_params


def read_shared_memory():
    return np.random.rand(256, 256, 3)


def initialize_shm(nbytes):

    fields_shm = create_shm(FIELDS_BUFFER_NAME, nbytes)
    shm_params = create_shm_params()
    file_path_shm = create_shm(FILES_BUFFER_NAME, MAX_FILEPATH_SIZE)

    return fields_shm, shm_params, file_path_shm