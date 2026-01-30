# tracking/kalman.py
from __future__ import annotations

import numpy as np


class KalmanFilter2D:
    """
    상태: [x, y, vx, vy]
    관측: [x, y]
    """
    def __init__(self, x: float, y: float, q: float = 2.0, r: float = 25.0):
        self.x = np.array([x, y, 0.0, 0.0], dtype=np.float32)

        self.P = np.eye(4, dtype=np.float32) * 100.0
        self.Q = np.eye(4, dtype=np.float32) * q
        self.R = np.eye(2, dtype=np.float32) * r

        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]], dtype=np.float32)

    def set_measurement_noise(self, r: float):
        """센서별 관측 노이즈 설정."""
        self.R = np.eye(2, dtype=np.float32) * r

    def predict(self, dt: float = 1.0):
        F = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]], dtype=np.float32)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q

    def update(self, zx: float, zy: float):
        z = np.array([zx, zy], dtype=np.float32)
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(4, dtype=np.float32)
        self.P = (I - K @ self.H) @ self.P

    def get_position(self) -> tuple[float, float]:
        return float(self.x[0]), float(self.x[1])

    def get_velocity(self) -> tuple[float, float]:
        return float(self.x[2]), float(self.x[3])

    def decay_velocity(self, factor: float = 0.5):
        """predict-only 반복 시 속도를 감쇠하여 무한 드리프트 방지."""
        self.x[2] *= factor
        self.x[3] *= factor
