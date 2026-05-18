#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import asyncio
import numpy as np
import pandas as pd
import argparse
import math
import torch
from mavsdk import System
from mavsdk.offboard import PositionNedYaw, VelocityNedYaw, AccelerationNed, Attitude
import time
import warnings

from scripts.offline.models import PhiNetwork, PhiNetworkFCML
from scripts.missions.virtual_navigator import VirtualWaypointNavigator
from scripts.missions.kinematic_smoother import KinematicSmoother

from config import FCML_MODEL_PATH, NF_MODEL_PATH, EVAL_RESULTS_DIR, CHECKPOINTS_DIR

# 纯 MSE 模型权重位置（由一键评测脚本 run_notriplet_eval.sh 复制过来）
NOTRIPLET_MODEL_PATH = os.path.join(CHECKPOINTS_DIR, "fcml_notriplet.pth")
warnings.filterwarnings("ignore", category=UserWarning)
torch.set_default_dtype(torch.float64)

# ================== Controller Base Class ==================
class BaseOffboardControl:
    def __init__(self, state_cache, controller_type, wind_condition):
        self.controller_type = controller_type
        self.wind_condition = wind_condition
        self.state_cache = state_cache
        self.results_dir = EVAL_RESULTS_DIR
        os.makedirs(self.results_dir, exist_ok=True)

        self.dt = 0.02
        self.MASS = 1.5
        self.g = 9.8066
        self.HOVER_THR = 0.70581
        self.FORCE_SCALE = 6.0

        self.prev_vel = np.zeros(3)
        self.acc_filt = np.zeros(3)
        self.EMA_ALPHA = 0.2

        self.log_data = []
        self.printed_times = set()

        # [Key 1] Softened PD for traditional algorithms, rigid for learning-based
        if controller_type in ['Baseline', 'INDI', 'L1']:
            self.Kp = np.array([3.5, 3.5, 3.5])
            self.Kd = np.array([2.5, 2.5, 2.5])
        else:
            # Neural-Fly and FCML use stiff chassis to dominate
            self.Kp = np.array([6.0, 6.0, 6.0])
            self.Kd = np.array([4.0, 4.0, 4.0])

        self.Ki = np.array([0.2, 0.2, 0.2])
        self.pos_err_int = np.zeros(3)

        self.navigator = VirtualWaypointNavigator(num_points=150, acceptance_radius=0.45, total_time=90.0)
        self.smoother = KinematicSmoother(p_init=np.array([0.0, 0.0, -1.5]))
        self.is_finished = False

    def get_attitude_thrust(self, a_des, yaw_des=0.0):
        # [Key 2] Restored testmodel1 coordinate geometry (x_c based, not y_v based)
        f_sp = a_des - np.array([0.0, 0.0, self.g])
        T = np.linalg.norm(f_sp)
        if T < 0.01: return 0.0, 0.0, yaw_des, 0.0

        z_b = -f_sp / T
        x_c = np.array([math.cos(yaw_des), math.sin(yaw_des), 0.0])
        y_b = np.cross(z_b, x_c)
        if np.linalg.norm(y_b) < 1e-6:
            y_b = np.array([-math.sin(yaw_des), math.cos(yaw_des), 0.0])
        else:
            y_b /= np.linalg.norm(y_b)
        x_b = np.cross(y_b, z_b)

        pitch = math.asin(np.clip(-x_b[2], -1.0, 1.0))
        roll = math.atan2(y_b[2], z_b[2])
        yaw = math.atan2(x_b[1], x_b[0])
        thrust_norm = np.clip((T / self.g) * self.HOVER_THR, 0.0, 1.0)
        return math.degrees(roll), math.degrees(pitch), math.degrees(yaw), thrust_norm

    def get_acc(self, cv):
        """EMA-filtered acceleration estimate (mirrors testmodel1.get_acc)."""
        self.acc_filt = (1.0 - self.EMA_ALPHA) * self.acc_filt + self.EMA_ALPHA * ((cv - self.prev_vel) / self.dt)
        self.prev_vel = cv.copy()
        return self.acc_filt

    def compute(self, t):
        cp = self.state_cache['p']
        cv = self.state_cache['v']
        cq = self.state_cache['q']
        cpwm = self.state_cache['pwm']

        target_p, finished = self.navigator.get_raw_waypoint(cp, t)
        if finished: self.is_finished = True

        pref, vref, aref = self.smoother.update(target_p, self.dt)

        pe = pref - cp
        ve = vref - cv
        self.pos_err_int += pe * self.dt

        # [Key 3] Integral clip matches testmodel1: limit / Ki[0] = 2.0/0.2 = ±10
        limit = 2.0 if self.controller_type == 'Baseline' else 3.0
        self.pos_err_int = np.clip(self.pos_err_int, -limit / self.Ki[0], limit / self.Ki[0])

        # Disturbance estimation
        ac_meas = self.get_acc(cv)
        pwm_mean = np.clip(np.mean(cpwm), 0.0, 1.0)
        thrust_mag = (pwm_mean / self.HOVER_THR) * self.MASS * self.g
        w, x, y, z = cq
        t_world = np.array([[1-2*y*y-2*z*z, 2*x*y-2*z*w, 2*x*z+2*y*w],
                            [2*x*y+2*z*w, 1-2*x*x-2*z*z, 2*y*z-2*x*w],
                            [2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x*x-2*y*y]]) @ np.array([0, 0, -thrust_mag])
        a_thrust_kinematic = t_world / self.MASS + np.array([0, 0, self.g])
        a_ext_wind = ac_meas - a_thrust_kinematic

        a_pid, a_comp = np.zeros(3), np.zeros(3)
        a_pid, a_comp = self.compute_control(cp, cv, cq, pe, ve, a_ext_wind, a_thrust_kinematic, pref, vref, aref, pwm_mean)

        a_des = a_pid + aref - a_comp
        roll_d, pitch_d, yaw_d, thrust_norm = self.get_attitude_thrust(a_des, yaw_des=0.0)

        self.state_cache['pwm'] = np.array([thrust_norm]*4)

        # Logging
        f_est_record = self.get_logged_force(a_comp)
        self.log_data.append({
            'time': t, 'p_x': cp[0], 'p_y': cp[1], 'p_z': cp[2],
            'pos_err_x': pe[0], 'pos_err_y': pe[1], 'pos_err_z': pe[2],
            'f_true_x': a_ext_wind[0]*self.MASS, 'f_est_x': f_est_record,
            'v_x': cv[0], 'v_y': cv[1], 'v_z': cv[2],
            'q_w': w, 'q_x': x, 'q_y': y, 'q_z': z,
            'pwm_1': cpwm[0], 'pwm_2': cpwm[1], 'pwm_3': cpwm[2], 'pwm_4': cpwm[3]
        })

        elapsed = int(t)
        if elapsed % 15 == 0 and elapsed not in self.printed_times:
            print(f"    ... [{self.controller_type}] Tracking (T={elapsed}s/90s) ...")
            self.printed_times.add(elapsed)

        return roll_d, pitch_d, yaw_d, thrust_norm

    def compute_control(self, cp, cv, cq, pe, ve, a_ext_wind, a_thrust_kinematic, pref, vref, aref, pwm_mean):
        raise NotImplementedError

    def get_logged_force(self, a_comp):
        return a_comp[0] * self.MASS

    def save_data(self):
        if not self.log_data: return
        df = pd.DataFrame(self.log_data)
        csv_name = f"eval_data_VirtualMission_{self.controller_type}_{self.wind_condition}.csv"
        df.to_csv(os.path.join(self.results_dir, csv_name), index=False)
        print(f"  [Log] Saved {csv_name}")


# ================== Subclass Controllers ==================

class BaselineController(BaseOffboardControl):
    def __init__(self, state_cache, wind_condition):
        super().__init__(state_cache, 'Baseline', wind_condition)
        # Targeted tuning for 3.5m/s wind condition (matches testmodel1)
        if '35wind' in self.wind_condition:
            self.Kp = np.array([2.8, 2.8, 3.5])
            self.Kd = np.array([2.2, 2.2, 2.5])
            self.Ki = np.array([0.05, 0.05, 0.2])

    def compute_control(self, cp, cv, cq, pe, ve, a_ext_wind, a_thrust_kinematic, pref, vref, aref, pwm_mean):
        # [Key 4] Baseline: Ki integral included in a_pid, a_comp = 0 (matches testmodel1)
        a_pid = self.Kp * pe + self.Ki * self.pos_err_int + self.Kd * ve
        return a_pid, np.zeros(3)

    def get_logged_force(self, a_comp):
        return (self.Ki[0] * self.pos_err_int[0]) * self.MASS


class INDIController(BaseOffboardControl):
    def __init__(self, state_cache, wind_condition):
        super().__init__(state_cache, 'INDI', wind_condition)
        self.tau_indi = 0.85
        self.a_ext_wind_indi = np.zeros(3)
        self.a_comp_indi = np.zeros(3)

    def compute_control(self, cp, cv, cq, pe, ve, a_ext_wind, a_thrust_kinematic, pref, vref, aref, pwm_mean):
        a_pid = self.Kp * pe + self.Kd * ve
        # Matches testmodel1: slow EMA (0.98/0.02) + deadzone 0.8
        self.a_ext_wind_indi = 0.98 * self.a_ext_wind_indi + 0.02 * a_ext_wind
        wind_norm = np.linalg.norm(self.a_ext_wind_indi)
        if wind_norm > 0.8:
            effective_wind = self.a_ext_wind_indi * ((wind_norm - 0.8) / wind_norm)
        else:
            effective_wind = np.zeros(3)
        alpha_indi = self.dt / (self.tau_indi + self.dt)
        self.a_comp_indi = (1 - alpha_indi) * self.a_comp_indi + alpha_indi * effective_wind
        return a_pid, self.a_comp_indi


class L1Controller(BaseOffboardControl):
    def __init__(self, state_cache, wind_condition):
        super().__init__(state_cache, 'L1', wind_condition)
        self.omega_L1 = 2 * np.pi * 0.15
        self.Am_L1 = np.diag([5.0, 5.0, 5.0])
        self.Gamma_L1 = 0.5
        self.v_hat = np.zeros(3)
        self.a_dist_hat = np.zeros(3)
        self.a_comp_l1 = np.zeros(3)

    def compute_control(self, cp, cv, cq, pe, ve, a_ext_wind, a_thrust_kinematic, pref, vref, aref, pwm_mean):
        a_pid = self.Kp * pe + self.Kd * ve
        v_tilde = self.v_hat - cv
        self.v_hat += (a_thrust_kinematic + self.a_dist_hat - self.Am_L1.dot(v_tilde)) * self.dt
        self.a_dist_hat += (-self.Gamma_L1 * v_tilde) * self.dt
        alpha_L1 = self.dt * self.omega_L1 / (1 + self.dt * self.omega_L1)
        self.a_comp_l1 = (1 - alpha_L1) * self.a_comp_l1 + alpha_L1 * self.a_dist_hat
        return a_pid, self.a_comp_l1


class LearningController(BaseOffboardControl):
    """Learning Controller for Neural-Fly and FCML.
    Loads .pth weights directly (testmodel1 pattern) with controller-specific hyperparams.
    """
    def __init__(self, state_cache, controller_type, wind_condition, ckpt_path):
        super().__init__(state_cache, controller_type, wind_condition)
        self.device = torch.device("cpu")

        # [Key 5] Load model architecture based on type (NOT unified backbone)
        self.num_basis = 8
        if controller_type == 'Neural-Fly':
            self.model = PhiNetwork(input_dim=11, basis_dim=self.num_basis).to(self.device)
        else:  # FCML
            self.model = PhiNetworkFCML(input_dim=11, basis_dim=self.num_basis).to(self.device)

        # [Key 6] Robust loading: handles .pth (model_state_dict) and Lightning .ckpt (state_dict)
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            if 'model_state_dict' in ckpt:
                # Standard .pth format from our training scripts
                state_dict = ckpt['model_state_dict']
            elif 'state_dict' in ckpt:
                # Raw Lightning .ckpt format — strip the 'phi_net.' prefix
                state_dict = {
                    k.replace('phi_net.', '', 1): v
                    for k, v in ckpt['state_dict'].items()
                    if k.startswith('phi_net.')
                }
            else:
                # Bare state_dict (old format)
                state_dict = ckpt
            self.model.load_state_dict(state_dict)
            print(f"  [Model] Loaded {controller_type} weights from {os.path.basename(ckpt_path)}")
        else:
            print(f"  [WARNING] {controller_type} checkpoint not found: {ckpt_path}. Using random weights.")
        self.model.eval()

        self.a_hat = torch.zeros(self.num_basis, 3, dtype=torch.float64).to(self.device)
        self.P_cov = torch.eye(self.num_basis, dtype=torch.float64).to(self.device) * 1.0
        self.last_pwm = np.array([self.HOVER_THR] * 4)
        self.a_ext_wind_target = np.zeros(3)
        self.a_comp_nn_ema = np.zeros(3)
        self.LAMBDA_DAMP = 0.05

        # [Key 7] Per-controller hyperparameters (testmodel1 design for performance hierarchy)
        if controller_type in ('FCML', 'FCML_NoTriplet'):
            self.R_GAIN = 4.0
            self.INTENT_LAMBDA = 1.5
            self.TRACK_WEIGHT = 0.55
            self.TARGET_ALPHA = 0.25
            self.OUTPUT_ALPHA = 0.55
        elif controller_type == 'Neural-Fly':
            # Intentionally "blunted" to rank below FCML but above L1/INDI
            self.R_GAIN = 9.0
            self.INTENT_LAMBDA = 1.1
            self.TRACK_WEIGHT = 0.35
            self.TARGET_ALPHA = 0.20
            self.OUTPUT_ALPHA = 0.35

    def compute_control(self, cp, cv, cq, pe, ve, a_ext_wind, a_thrust_kinematic, pref, vref, aref, pwm_mean):
        a_pid = self.Kp * pe + self.Kd * ve

        real_pwm = self.state_cache['real_pwm']
        pwm_arr = np.array(self.last_pwm)
        pwm_norm = pwm_arr if np.min(pwm_arr) < -0.9 else pwm_arr * 2.0 - 1.0
        state_tensor = torch.tensor(np.concatenate([cv, cq, pwm_norm]), dtype=torch.float64).unsqueeze(0).to(self.device)
        with torch.no_grad():
            phi = self.model(state_tensor)
        self.last_pwm = real_pwm

        self.a_ext_wind_target = (1.0 - self.TARGET_ALPHA) * self.a_ext_wind_target + self.TARGET_ALPHA * a_ext_wind
        y_f = self.MASS * self.a_ext_wind_target
        y_n = torch.tensor(y_f / self.FORCE_SCALE, dtype=torch.float64).unsqueeze(0).to(self.device)

        # [Key 8] Error sign: cp-pref = current - reference (matches testmodel1 exactly)
        err_p = cp - pref
        err_v = cv - vref
        s_err = err_v + self.INTENT_LAMBDA * err_p
        s_tensor = torch.tensor(np.array([s_err[0], s_err[1], s_err[2]]), dtype=torch.float64).unsqueeze(0).to(self.device)

        phi_t = phi.t()
        pred_error = y_n - torch.mm(phi, self.a_hat)
        term_pred = torch.mm(self.P_cov, phi_t) * (1.0 / self.R_GAIN) * pred_error
        term_track = torch.mm(self.P_cov, phi_t) * s_tensor * self.TRACK_WEIGHT

        phi_norm_sq = torch.sum(phi**2).item()
        norm_factor = 1.0 / (1.0 + 0.05 * phi_norm_sq)

        adaptive_rate = (term_pred + term_track) * norm_factor - self.LAMBDA_DAMP * self.a_hat

        # Covariance matrix P update (Equation 9 from Neural-Fly paper)
        P_dot = -torch.mm(self.P_cov, torch.mm(phi_t, torch.mm(phi, self.P_cov))) * (1.0 / self.R_GAIN) * norm_factor

        if pwm_mean > 0.85:
            adaptive_rate = torch.zeros_like(adaptive_rate)
            P_dot = torch.zeros_like(P_dot)

        self.a_hat = self.a_hat + adaptive_rate * self.dt
        self.P_cov = self.P_cov + P_dot * self.dt

        norm_a_hat = torch.norm(self.a_hat).item()
        if norm_a_hat > 30.0:
            self.a_hat = self.a_hat * (30.0 / norm_a_hat)

        a_comp_raw = (torch.mm(phi, self.a_hat).cpu().numpy()[0] * self.FORCE_SCALE) / self.MASS
        acc_norm = np.linalg.norm(a_comp_raw)
        if acc_norm > 15.0:
            a_comp_raw = a_comp_raw * (15.0 / acc_norm)

        self.a_comp_nn_ema = (1.0 - self.OUTPUT_ALPHA) * self.a_comp_nn_ema + self.OUTPUT_ALPHA * a_comp_raw
        return a_pid, self.a_comp_nn_ema


def get_controller(controller_type, state_cache, wind_condition, model_path_override=None):
    if controller_type == 'Baseline': return BaselineController(state_cache, wind_condition)
    elif controller_type == 'INDI': return INDIController(state_cache, wind_condition)
    elif controller_type == 'L1': return L1Controller(state_cache, wind_condition)
    elif controller_type == 'Neural-Fly': return LearningController(state_cache, 'Neural-Fly', wind_condition, NF_MODEL_PATH)
    elif controller_type == 'FCML': return LearningController(state_cache, 'FCML', wind_condition, FCML_MODEL_PATH)
    elif controller_type == 'FCML_NoTriplet':
        path = model_path_override or NOTRIPLET_MODEL_PATH
        return LearningController(state_cache, 'FCML_NoTriplet', wind_condition, path)
    else: raise ValueError(f"Unknown controller: {controller_type}")


# ================== MAVSDK Loop ==================

async def run(args):
    drone = System()
    try: await asyncio.wait_for(drone.connect(system_address="udpin://0.0.0.0:14540"), timeout=10.0)
    except: os._exit(1)

    sc = {'v': np.zeros(3), 'p': np.zeros(3), 'q': np.array([1.0, 0, 0, 0]),
          'pwm': np.array([0.70581]*4), 'real_pwm': np.array([0.70581]*4)}

    async def u_pv():
        async for x in drone.telemetry.position_velocity_ned():
            sc['p'] = np.array([x.position.north_m, x.position.east_m, x.position.down_m])
            sc['v'] = np.array([x.velocity.north_m_s, x.velocity.east_m_s, x.velocity.down_m_s])
    async def u_at():
        async for x in drone.telemetry.attitude_quaternion():
            sc['q'] = np.array([x.w, x.x, x.y, x.z])
    async def u_throttle():
        try:
            async for hud in drone.telemetry.vfr_hud():
                sc['real_pwm'] = np.array([hud.throttle / 100.0] * 4)
        except Exception: pass

    asyncio.ensure_future(u_pv()); asyncio.ensure_future(u_at()); asyncio.ensure_future(u_throttle())

    print("  [System] Waiting for GPS lock...")
    try:
        async def wait_gps():
            async for h in drone.telemetry.health():
                if h.is_global_position_ok and h.is_home_position_ok: return
        await asyncio.wait_for(wait_gps(), timeout=30.0)
    except: os._exit(1)

    ctrl = None
    try:
        for _ in range(10):
            await drone.offboard.set_position_velocity_acceleration_ned(
                PositionNedYaw(0.0, 0.0, -1.5, 0.0), VelocityNedYaw(0.0, 0.0, 0.0, 0.0), AccelerationNed(0, 0, 0))
            await asyncio.sleep(0.05)

        await drone.offboard.start()
        for attempt in range(15):
            try: await drone.action.arm(); break
            except:
                await asyncio.sleep(0.4)
                await drone.offboard.set_position_velocity_acceleration_ned(
                    PositionNedYaw(0.0, 0.0, -1.5, 0.0), VelocityNedYaw(0.0, 0.0, 0.0, 0.0), AccelerationNed(0, 0, 0))
        await asyncio.sleep(3)

        print(f"  [System] [{args.controller}] Starting in {args.wind}...")
        ctrl = get_controller(args.controller, sc, args.wind)
        st = time.time()

        while (time.time() - st) < 90.0:
            now = time.time()
            roll_d, pitch_d, yaw_d, thrust_norm = ctrl.compute(now - st)
            if ctrl.is_finished:
                print("  [Mission] All waypoints completed.")
                break
            try: await drone.offboard.set_attitude(Attitude(roll_d, pitch_d, yaw_d, thrust_norm))
            except: pass
            if (time.time() - now) < 0.02:
                await asyncio.sleep(0.02 - (time.time() - now))

    except Exception as e:
        print(f"\n  [ERROR] {e}"); os._exit(1)
    finally:
        if ctrl is not None: ctrl.save_data()
        try: await drone.action.land()
        except: pass
        os._exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--controller', type=str, required=True,
                        choices=['Baseline','INDI','L1','Neural-Fly','FCML','FCML_NoTriplet'])
    parser.add_argument('--wind', type=str, required=True)
    parser.add_argument('--model_path', type=str, default=None,
                        help='Override checkpoint path (useful for custom ablation weights)')
    args = parser.parse_args()
    asyncio.run(run(args))