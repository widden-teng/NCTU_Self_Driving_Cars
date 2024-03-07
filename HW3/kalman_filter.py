import numpy as np


class KalmanFilter():
    def __init__(self, x=0, y=0, yaw=0):
        # State [x, y, yaw]
        self.x = np.array([x, y, yaw])
        # Transition matrix
        self.A = np.identity(3)
        self.B = np.identity(3)
        # Error matrix
        self.P = np.identity(3) * 1
        # Observation matrix
        self.H = np.array([[1, 0, 0],
                           [0, 1, 0]])
        # State transition error covariance
        self.Q = np.array([[0.5, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, 0.2]])
        # Measurement error
        # self.R = np.array([0.75]).reshape(1, 1)
        self.R = np.array([[0.53, 0], [0, 0.82]])

    def predict(self, u):
        self.x = np.dot(self.A, self.x) + np.dot(self.B, u)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q

    def update(self, z):
        y = z - np.dot(self.H, self.x)
        S = self.R + np.dot(self.H, np.dot(self.P, self.H.T))
        # find inverse matrix
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        I = np.eye(3)
        self.P = np.dot(I - np.dot(K, self.H), self.P)
        return self.x, self.P
