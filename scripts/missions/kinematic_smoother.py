import numpy as np

class KinematicSmoother:
    def __init__(self, p_init):
        self.p = np.array(p_init, dtype=float)
        self.v = np.zeros(3)
        self.a = np.zeros(3)
        self.kp = 9.0  
        self.kd = 6.0  
        self.max_v = 3.5
        self.max_a = 6.0

    def update(self, target_p, dt):
        a_des = self.kp * (target_p - self.p) + self.kd * (np.zeros(3) - self.v)
        a_des = np.clip(a_des, -self.max_a, self.max_a)
        
        self.v += a_des * dt
        v_norm = np.linalg.norm(self.v)
        if v_norm > self.max_v:
            self.v = (self.v / v_norm) * self.max_v
            
        self.p += self.v * dt
        self.a = a_des
        return self.p.copy(), self.v.copy(), self.a.copy()
