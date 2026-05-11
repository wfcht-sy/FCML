#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import asyncio
import math
import subprocess
import sys
import os
import time
from typing import Tuple
from mavsdk import System
from mavsdk.offboard import OffboardError, PositionNedYaw

FLIGHT_DURATION_SEC = 90
CENTER_ALTITUDE_METERS = -1.5

class LemniscateTrajectory:
    def __init__(self, mission_start_time_sec: float):
        self.mission_start_time_sec = mission_start_time_sec

    def get_target_pose(self, current_system_time_sec: float) -> Tuple[float, float, float, float]:
        t = current_system_time_sec - self.mission_start_time_sec
        target_x = 1.25 * math.sin(t)
        target_y = 0.0
        target_z = 0.75 * math.sin(2 * t) + CENTER_ALTITUDE_METERS
        return target_x, target_y, target_z, 0.0

async def execute_flight_mission(wind_parameters: list) -> None:
    drone = System()
    print("  [System] Connecting to MAVSDK (udpin://0.0.0.0:14540)...")
    await drone.connect(system_address="udpin://0.0.0.0:14540")
    async for state in drone.core.connection_state():
        if state.is_connected: break

    print("  [System] Waiting for GPS lock...")
    async for health in drone.telemetry.health():
        if health.is_global_position_ok and health.is_home_position_ok: break
        await asyncio.sleep(2)

    subprocess.run(["bash", "./set_wind.sh"] + wind_parameters, check=False)

    print("  [System] Arming and taking off...")
    try: await drone.action.arm()
    except Exception: return

    await drone.action.set_takeoff_altitude(abs(CENTER_ALTITUDE_METERS))
    await drone.action.takeoff()

    async for position in drone.telemetry.position():
        if position.relative_altitude_m > 1.4: break
    await asyncio.sleep(2)

    initial_setpoint = PositionNedYaw(0.0, 0.0, CENTER_ALTITUDE_METERS, 0.0)
    await drone.offboard.set_position_ned(initial_setpoint)

    try: await drone.offboard.start()
    except OffboardError: return

    print(f"  [Mission] Starting vertical lemniscate waypoint flight ({FLIGHT_DURATION_SEC} s)...")
    trajectory_generator = LemniscateTrajectory(time.time())
    mission_start_time = time.time()
    time_step_sec = 0.02
    last_print_time = 0

    while time.time() - mission_start_time < FLIGHT_DURATION_SEC:
        elapsed = time.time() - mission_start_time
        if time.time() - last_print_time > 5.0:
            print(f"    -> Flight progress: {elapsed:.1f} / {FLIGHT_DURATION_SEC} s")
            last_print_time = time.time()

        target_x, target_y, target_z, target_yaw = trajectory_generator.get_target_pose(time.time())
        await drone.offboard.set_position_ned(PositionNedYaw(target_x, target_y, target_z, target_yaw))
        await asyncio.sleep(time_step_sec)

    print("  [System] Mission time up, landing...")
    try: await drone.offboard.stop()
    except Exception: pass
    await drone.action.land()

    # Prevent deadlock
    print("  [System] Waiting for touchdown and auto-disarm...")
    async for is_armed in drone.telemetry.armed():
        if not is_armed:
            print("  [System] Successfully disarmed. Exiting safely.")
            break

    os._exit(0) # Force exit to prevent gRPC thread deadlocks

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("wind", nargs=6)
    arguments = parser.parse_args()
    try: asyncio.run(execute_flight_mission(arguments.wind))
    except KeyboardInterrupt: pass
