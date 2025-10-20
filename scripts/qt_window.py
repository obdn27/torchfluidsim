from collections import deque
import sys, os
import threading, time
from datetime import datetime

import numpy as np
from PyQt6.QtWidgets import (
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QSlider,
    QLabel,
    QSizePolicy,
    QApplication,
    QPushButton,
    QFileDialog,
)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage
import pyqtgraph as pg
import matplotlib.pyplot as plt

from config import *
from sim_stepper import SimulationStepper
from shm_manager import SharedMemoryManager


class InteractiveLabel(QLabel):
    # Signal to emit mouse coordinates relative to this label.
    mouse_position_signal = pyqtSignal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)

    def mouseMoveEvent(self, event):
        # Get the mouse position relative to this label.
        pos = event.pos()
        self.mouse_position_signal.emit(pos.x(), pos.y())
        super().mouseMoveEvent(event)


class VisualisationThread(QThread):
    frame_signal = pyqtSignal(np.ndarray)

    def __init__(self, shm_manager, parent=None):
        super().__init__(parent)
        self.shm_manager = shm_manager

    def run(self):
        while True:
            frame = self.shm_manager.read_fields()  # returns a copy of the fields buffer
            self.frame_signal.emit(frame)
            time.sleep(1 / 60)


class SimulationUI(QMainWindow):
    def __init__(self, shm_manager):
        super().__init__()
        self.shm_manager = shm_manager
        self.setGeometry(100, 100, 1200, 600)
        self.centralWidget = QWidget()
        self.setCentralWidget(self.centralWidget)
        self.current_width = SIM_PARAMS_DEFAULTS["grid_width"][0]

        main_layout = QHBoxLayout()

        # Left control panel
        self.control_panel = QWidget()
        control_layout = QVBoxLayout(self.control_panel)
        self.sliders = {}
        # Define parameters that you do not want to create sliders for.
        skip_params = [
            "mouse_x", "mouse_y", "density_scaling", "dx", "dy",
            "reset_request", "obstacle_path", "interaction_strength", "interaction_radius"
        ]

        # Loop over all parameters in SIM_PARAMS_DEFAULTS.
        for param_name, (default_value, minimum, maximum, step) in SIM_PARAMS_DEFAULTS.items():
            if param_name in skip_params:
                continue

            slider_layout = QVBoxLayout()
            label = QLabel(f"{param_name}: {default_value:.2f}")
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setMinimum(int(minimum / step))
            slider.setMaximum(int(maximum / step))
            slider.setValue(int(default_value / step))
            slider.setSingleStep(int(step / step))
            slider.setFixedWidth(300)
            slider.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.MinimumExpanding)

            # Bind the slider change event.
            slider.valueChanged.connect(
                lambda val, p=param_name, lbl=label, s=step: self._slider_changed(val, p, lbl, s)
            )
            slider_layout.addWidget(label)
            slider_layout.addWidget(slider)
            control_layout.addLayout(slider_layout)
            self.sliders[param_name] = slider

        # Obstacle texture file picker.
        self.file_picker_button = QPushButton("Load Obstacle Texture")
        self.file_picker_button.clicked.connect(self.get_obstacle_path)
        control_layout.addWidget(self.file_picker_button)

        # Visualisation feed display.
        self.visualisation_label = InteractiveLabel()
        self.visualisation_label.setFixedSize(512, 512)
        # Connect the signal to the mouse position update slot.
        self.visualisation_label.mouse_position_signal.connect(self.update_mouse_position)

        main_layout.addWidget(self.control_panel)
        main_layout.addWidget(self.visualisation_label)

        # Right charts panel.
        self.charts_panel = QWidget()
        charts_layout = QVBoxLayout(self.charts_panel)
        self.charts = []
        for _ in range(6):
            plot = pg.PlotWidget()
            plot.setYRange(0, 1)
            self.charts.append(plot)
            charts_layout.addWidget(plot)
        main_layout.addWidget(self.charts_panel)

        self.centralWidget.setLayout(main_layout)

        self.metric_series = [deque(maxlen=100) for _ in range(6)]
        self.curve = [plot.plot([]) for plot in self.charts]

        # Start the visualisation thread.
        self.vis_thread = VisualisationThread(self.shm_manager)
        self.vis_thread.frame_signal.connect(self.update_visualisation_feed)
        self.vis_thread.start()

        # Set up a timer for updating charts.
        self.graph_timer = QTimer()
        self.graph_timer.timeout.connect(self.update_graphs)
        self.graph_timer.start(1000 // 60)

        self.store = np.zeros(shape=(6, MAX_METRICS_LEN))
        self.n_frame = 0

        self.setMouseTracking(True)

    def update_mouse_position(self, x, y):
        x, y = y, x
        # Get the dimensions of the label (which are fixed at 512).
        label_width = self.visualisation_label.width()
        label_height = self.visualisation_label.height()\
        # Scale x and y relative to the current grid width.
        grid_x = (x / label_width) * self.current_width
        grid_y = (y / label_height) * self.current_width  # assuming a square grid

        # Update the shared memory manager parameters.
        self.shm_manager.update_param("mouse_x", float(grid_x))
        self.shm_manager.update_param("mouse_y", float(grid_y))

    def _slider_changed(self, value, param, label, step):

        mapping = {
            0.0: "ink density",
            1.0: "horizontal velocity",
            2.0: "vertical velocity",
            3.0: "divergence",
            4.0: "pressure",
            5.0: "obstacle texture",
        }

        float_value = value * step
        label.setText(f"{param}: {float_value:.2f}")
        if param == "grid_width":
            self.current_width = int(float_value)
        elif param == "current_field":
            label.setText(f"{param}: {mapping[value]}")

        self.update_param(param, float_value)
        # For debugging: print current field from shared memory.
        grid_width = self.shm_manager.read_params()[SIM_PARAMS["grid_width"]]
        print(f"grid width: {grid_width}")

    def update_param(self, param_name, val):
        self.shm_manager.update_param(param_name, float(val))

    def get_obstacle_path(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Obstacle Texture", "", "Image Files (*.png *.jpg *.jpeg)"
        )
        if file_path:
            self.shm_manager.set_file_path(file_path)
            print(f"Loaded Obstacle Texture: {file_path}")

        # Reset store to eliminate irrelevant scalar metrics
        self.store[...] = 0

    def update_visualisation_feed(self, frame):
        # Crop the frame to the current grid size.
        cropped = frame[:self.current_width, :self.current_width, :]
        H, W, C = cropped.shape  # H and W should equal self.current_width
        bytes_per_line = C * W
        frame_uint8 = (cropped * 255).astype(np.uint8)
        frame_uint8_transposed = np.ascontiguousarray(np.transpose(frame_uint8, (1, 0, 2)))
        q_img = QImage(frame_uint8_transposed.data, W, H, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img).scaled(512, 512, Qt.AspectRatioMode.KeepAspectRatio)
        self.visualisation_label.setPixmap(pixmap)

    def update_graphs(self):
        data = self.shm_manager.read_fields()[:self.current_width, :self.current_width, :]
        xvel = data[:, :, 0]
        yvel = data[:, :, 1]
        pressure = data[:, :, 2]

        metrics = [
            float(xvel.mean()),
            float(xvel.std()),
            float(yvel.mean()),
            float(yvel.std()),
            float(pressure.mean()),
            float(pressure.std()),
        ]

        for i in range(6):
            self.metric_series[i].append(metrics[i])
            self.curve[i].setData(list(self.metric_series[i]))
            self.charts[i].setYRange(0, max(self.metric_series[i]) * 1.5)

        self.n_frame += 1
        self.store[:, self.n_frame % MAX_METRICS_LEN] = np.array(metrics)


def save_to_disk(timeseries_data):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_log_dir = "logs"
    
    log_dir = os.path.join(base_log_dir, f"log_{timestamp}")
    os.makedirs(log_dir, exist_ok=False)

    # Save as compressed NPZ file.
    npz_path = os.path.join(log_dir, "data.npz")
    np.savez_compressed(npz_path, data_array=timeseries_data)
    
    # Save as CSV file.
    csv_path = os.path.join(log_dir, "data.csv")
    np.savetxt(csv_path, timeseries_data, delimiter=",", fmt="%s")


def exit_handler(*args):
    save_to_disk(window.store)


if __name__ == "__main__":
    # Initialize the shared memory manager.
    shm_manager_instance = SharedMemoryManager()

    # Create and start the simulation stepper in its own thread.
    sim_instance = SimulationStepper()
    sim_thread = threading.Thread(target=sim_instance.run, daemon=True)

    # sim_thread = threading.Thread(target=sim_stepper.test, daemon=True)
    sim_thread.start()

    app = QApplication(sys.argv)
    app.aboutToQuit.connect(exit_handler)
    window = SimulationUI(shm_manager_instance)
    window.setWindowTitle("PyTorch Fluid simulator")
    window.show()
    sys.exit(app.exec())
