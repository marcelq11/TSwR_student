import numpy as np
from .controller import Controller
from models.manipulator_model import ManiuplatorModel


class MMAController(Controller):
    def __init__(self, Tp):
        # TODO: Fill the list self.models with 3 models of 2DOF manipulators with different m3 and r3
        # I:   m3=0.1,  r3=0.05
        # II:  m3=0.01, r3=0.01
        # III: m3=1.0,  r3=0.3
        self.models = [ManiuplatorModel(Tp, 0.1, 0.05),
                       ManiuplatorModel(Tp, 0.01, 0.01),
                       ManiuplatorModel(Tp, 1.0, 0.3)]
        self.i = 0
        self.Tp = Tp
        self.prev_x = np.zeros(4)
        self.prev_u = np.zeros(2)

    def choose_model(self, x):
        # TODO: Implement procedure of choosing the best fitting model from self.models (by setting self.i) based on x
        x_mi = []
        for model in self.models:
            x_mi.append((model.x_dot(self.prev_x, self.prev_u) - self.prev_x.reshape(4, 1)) / self.Tp)
        pass
        x_reshaped = x.reshape(4, 1)
        errors = list(map( lambda x: np.sum(abs(x_reshaped - x)), x_mi))
        self.i = np.argmin(errors)
        pass

    def calculate_control(self, x, q_r, q_r_dot, q_r_ddot):
        self.choose_model(x)
        q = x[:2]
        q_dot = x[2:]
        e = q_r - q
        e_dot = q_r_dot - q_dot
        Kd = np.array([[25, 0], [0, 25]])
        Kp = np.array([[60, 0], [0, 60]])
        v = q_r_ddot + Kd @ e_dot + Kp @ e
        M = self.models[self.i].M(x)
        C = self.models[self.i].C(x)
        u = M @ v[:, np.newaxis] + C @ q_dot[:, np.newaxis]
        self.prev_x = x
        self.prev_u = u
        return u
