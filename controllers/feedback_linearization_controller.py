import numpy as np
from models.manipulator_model import ManiuplatorModel
from .controller import Controller


class FeedbackLinearizationController(Controller):
    def __init__(self, Tp):
        self.model = ManiuplatorModel(Tp)


    def calculate_control(self, x, q_r, q_r_dot, q_r_ddot):
        q1, q2, q1_dot, q2_dot = x

        e = q_r - [q1, q2]
        e_dot = q_r_dot - [q1_dot, q2_dot]

        Kd = np.array([[25, 0], [0, 25]])
        Kp = np.array([[60, 0], [0, 60]])

        v = q_r_ddot + Kd @ e_dot + Kp @ e

        tau = self.model.M(x) @ v[:, np.newaxis] + self.model.C(x) @ q_r_dot[:, np.newaxis]

        return tau
