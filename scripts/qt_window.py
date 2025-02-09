import sys, os
import numpy as np
import threading, subprocess
from multiprocessing import shared_memory
import time
import shm_manager
from sim_stepper import sim_stepper

from PyQt6.QtWidgets import (
    QMainWindow, 
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QListWidget,
    QSlider,
    QLabel,
    QGraphicsView,
    QGraphicsScene,
    QSizePolicy,
    QApplication,
)

from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage
import pyqtgraph as pg

from config import *


class VisualisationThread(QThread):
    frame_signal = pyqtSignal(np.ndarray)

    def run(self):
        while True:
            frame = shm_manager.read_shared_memory()
            self.frame_signal.emit(frame)
            time.sleep(1/60)


class ProjectUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setGeometry(100, 100, 1200, 600)

        self.centralWidget = QWidget()
        self.setCentralWidget(self.centralWidget)
        layout = QHBoxLayout()

        self.control_panel = QWidget()
        self.setCentralWidget(self.centralWidget)
        layout = QHBoxLayout()

        # Left sidebar
        self.control_panel = QWidget()
        control_layout = QVBoxLayout()
        self.control_panel.setLayout(control_layout)

        self.sliders = {}

        for param_name, (default_value, minimum, maximum, step) in SIM_PARAMS_DEFAULTS.items():

            if param_name in ["mouse_x", "mouse_y", "density_scaling", "dx", "dy", "reset_request", "obstacle_path", "grid_width", "interaction_strength", "interaction_radius"]:
                continue

            scale_factor = 100

            slider_layout = QVBoxLayout()

            label = QLabel(f"{param_name}: {default_value:.2f}")

            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setMinimum(int(minimum * scale_factor))
            slider.setMaximum(int(maximum * scale_factor))
            slider.setValue(int(default_value * scale_factor))
            slider.setSingleStep(int(step * scale_factor))

            slider.setFixedWidth(300)
            slider.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.MinimumExpanding)

            def slider_changed(value, param=param_name, scale=scale_factor, lbl=label):
                float_value = value / scale_factor
                lbl.setText(f"{param}: {float_value:.2f}")
                self.update_param(param, float_value)

            slider.valueChanged.connect(lambda val, p=param_name, lbl=label: slider_changed(val, p, lbl=lbl))

            slider_layout.addWidget(label)
            slider_layout.addWidget(slider)

            control_layout.addLayout(slider_layout)

            self.sliders[param_name] = slider


        # Visualisation feed
        self.visualisation_label = QLabel()
        self.visualisation_label.setFixedSize(512, 512)
        layout.addWidget(self.control_panel)
        layout.addWidget(self.visualisation_label)

        # Right charts
        self.charts_panel = QWidget()
        charts_layout = QVBoxLayout()
        self.charts_panel.setLayout(charts_layout)

        self.charts = []
        for _ in range(6):
            plot = pg.PlotWidget()
            plot.setYRange(0, 0.1)
            self.charts.append(plot)
            charts_layout.addWidget(plot)
        layout.addWidget(self.charts_panel)

        self.centralWidget.setLayout(layout)

        self.visualisation_thread = VisualisationThread()
        self.visualisation_thread.frame_signal.connect(self.update_visualisation_feed)
        self.visualisation_thread.start()

        self.graph_timer = QTimer()
        self.graph_timer.timeout.connect(self.update_graphs)
        self.graph_timer.start(1000 // 60)

    @staticmethod
    def update_param(param_name, value):
        """ Returns a function that updates the shared memory with the new parameter value. """
        return lambda val: shm_manager.update_simulation_param(param_name, float(val), shm_manager.create_shm_params())

    def update_visualisation_feed(self, frame):
        H, W, C = frame.shape
        bytes_per_line = C * W
        q_img = QImage(frame.data, W, H, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img).scaled(512, 512, Qt.AspectRatioMode.KeepAspectRatio)
        self.visualisation_label.setPixmap(pixmap)

    def update_graphs(self):
        """Update charts with real-time data."""
        # data = process_rgb_for_graphing(read_shared_memory())
        data = shm_manager.read_shared_memory()
        for i, plot in enumerate(self.charts):
            plot.plot([data[i % 3, i % 3, i % 3]], clear=True)


def halt_sim_stepper_thread(shm_params):
    shm_manager.update_simulation_param('run_flag', 1, shm_params)


print("qt_window.py running", __name__)

if __name__ == "__main__":

    fields_shm, shm_params, file_path_shm = shm_manager.initialize_shm(int(np.prod(MAX_RES) * 4 * 7))

    sim_thread = threading.Thread(target=sim_stepper, daemon=True)
    sim_thread.start()

    app = QApplication(sys.argv)
    window = ProjectUI()
    window.show()
    sys.exit(app.exec())

    halt_sim_stepper_thread(shm_params)

