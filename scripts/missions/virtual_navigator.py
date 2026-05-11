import numpy as np
import math

class VirtualWaypointNavigator:
    def __init__(self, num_points=150, acceptance_radius=0.45, total_time=90.0): 
        self.waypoints = []
        for i in range(num_points):
            theta = (i / (num_points - 1)) * 10 * math.pi
            x = 4.0 * math.sin(theta)
            y = 4.0 * math.sin(theta) * math.cos(theta)
            self.waypoints.append(np.array([x, y, -1.5]))
        self.current_idx = 0
        self.num_points = num_points
        self.acceptance_radius = acceptance_radius
        self.total_time = total_time

    def get_raw_waypoint(self, current_p, t):
        if self.current_idx >= self.num_points:
            return self.waypoints[-1], True
            
        # 1. Spatial proximity check
        target_p = self.waypoints[self.current_idx]
        if np.linalg.norm(target_p - current_p) < self.acceptance_radius:
            self.current_idx += 1
            
        # 2. Time-guaranteed progression (ensures global convergence)
        min_expected_idx = int((t / self.total_time) * self.num_points)
        if self.current_idx < min_expected_idx:
            self.current_idx = min_expected_idx
            
        if self.current_idx >= self.num_points:
            return self.waypoints[-1], True
            
        return self.waypoints[self.current_idx], False
