from collections import deque
import sys
import numpy as np
import threading, time
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
from config import *
import matplotlib.pyplot as plt

from sim_stepper import SimulationStepper
import sim_stepper
from shm_manager import SharedMemoryManager


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
        self.visualisation_label = QLabel()
        self.visualisation_label.setFixedSize(512, 512)
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

    def _slider_changed(self, value, param, label, step):
        float_value = value * step
        label.setText(f"{param}: {float_value:.2f}")
        if param == "grid_width":
            self.current_width = int(float_value)
        self.update_param(param, float_value)
        # For debugging: print current field from shared memory.
        current_field = self.shm_manager.read_params()[SIM_PARAMS["current_field"]]
        print(f"\t\t current field: {current_field}")

    def update_param(self, param_name, val):
        self.shm_manager.update_param(param_name, float(val))

    def get_obstacle_path(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Obstacle Texture", "", "Image Files (*.png *.jpg *.jpeg)"
        )
        if file_path:
            self.shm_manager.set_file_path(file_path)
            print(f"Loaded Obstacle Texture: {file_path}")

    def update_visualisation_feed(self, frame):
        H, W, C = frame.shape
        bytes_per_line = C * W
        frame_uint8 = (frame * 255).astype(np.uint8)
        frame_uint8_transposed = np.ascontiguousarray(np.transpose(frame_uint8, (1, 0, 2)))
        # Use the current_width for both dimensions.
        q_img = QImage(frame_uint8_transposed.data, self.current_width, self.current_width, bytes_per_line, QImage.Format.Format_RGB888)
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


if __name__ == "__main__":
    # Initialize the shared memory manager.
    shm_manager_instance = SharedMemoryManager()

    # Create and start the simulation stepper in its own thread.
    sim_instance = SimulationStepper()
    sim_thread = threading.Thread(target=sim_instance.run, daemon=True)

    # sim_thread = threading.Thread(target=sim_stepper.test, daemon=True)
    sim_thread.start()

    app = QApplication(sys.argv)
    window = SimulationUI(shm_manager_instance)
    window.show()
    sys.exit(app.exec())
