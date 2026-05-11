#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Native PX4 Waypoint Mission Test Script
Logic:
1. Tighten NAV_ACC_RAD to 1m
2. Extend trajectory to 5 loops, relax total flight time to 90 seconds.
"""

import asyncio
import numpy as np
import pandas as pd
import argparse
import math
from mavsdk import System
from mavsdk.mission import MissionItem, MissionPlan
import time
import os

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import EVAL_RESULTS_DIR
RESULTS_DIR = EVAL_RESULTS_DIR
os.makedirs(RESULTS_DIR, exist_ok=True)

class MissionBaseline:
    def __init__(self, wind_condition='0'):
        self.wind_condition = wind_condition
        self.log_data = []
        self.is_running = True

    def get_8_figure_waypoints(self, home_lat, home_lon, num_points=150):
        """Generate relative figure-8 trajectory waypoints around Home"""
        items = []
        # Extended trajectory to 5 loops (10 * pi)
        for i in range(num_points):
            theta = (i / (num_points - 1)) * 10 * math.pi
            x = 4.0 * math.sin(theta)
            y = 4.0 * math.sin(theta) * math.cos(theta)
            
            lat = home_lat + (x / 111319.5)
            lon = home_lon + (y / (111319.5 * math.cos(math.radians(home_lat))))
            
            items.append(MissionItem(
                latitude_deg=lat,
                longitude_deg=lon,
                relative_altitude_m=1.5,        
                speed_m_s=2.5,                  
                is_fly_through=True,            
                gimbal_pitch_deg=float('nan'),  
                gimbal_yaw_deg=float('nan'),    
                camera_action=MissionItem.CameraAction.NONE,
                loiter_time_s=float('nan'),     
                camera_photo_interval_s=float('nan'), 
                acceptance_radius_m=0.3,        # Tightened acceptance radius to 0.3m
                yaw_deg=float('nan'),           
                camera_photo_distance_m=float('nan'),
                vehicle_action=MissionItem.VehicleAction.NONE
            ))
        return items

    async def record_data(self, drone):
        """Asynchronously record high-precision local NED coordinates"""
        print("  [Recorder] Starting to collect 50Hz local coordinate data...")
        start_time = time.time()
        async for pos_vel in drone.telemetry.position_velocity_ned():
            if not self.is_running: break
            self.log_data.append({
                'time': time.time() - start_time,
                'p_x': pos_vel.position.north_m,
                'p_y': pos_vel.position.east_m,
                'p_z': pos_vel.position.down_m
            })
            await asyncio.sleep(0.02) 

    def save_data(self):
        if self.log_data:
            pd.DataFrame(self.log_data).to_csv(
                os.path.join(RESULTS_DIR, f"eval_data_Mission_{self.wind_condition}.csv"), index=False)

async def run(args):
    drone = System()
    await drone.connect(system_address="udpin://0.0.0.0:14540")

    print("  [System] Waiting for GPS lock...")
    async for health in drone.telemetry.health():
        if health.is_global_position_ok and health.is_home_position_ok: break

    try: await drone.telemetry.set_rate_position_velocity_ned(50.0)
    except: pass

    home_lat, home_lon = 0.0, 0.0
    async for pos in drone.telemetry.position():
        home_lat = pos.latitude_deg
        home_lon = pos.longitude_deg
        break

    print("  [Param] Forcing NAV_ACC_RAD to a strict 1m...")
    await drone.param.set_param_float("NAV_ACC_RAD", 1)

    mb = MissionBaseline(args.wind)
    
    mission_items = mb.get_8_figure_waypoints(home_lat, home_lon)
    mission_plan = MissionPlan(mission_items)
    await drone.mission.upload_mission(mission_plan)
    print(f"  [Mission] {len(mission_items)} dense waypoints generated and uploaded.")

    print("  [System] Taking off...")
    await drone.action.arm()
    await drone.action.takeoff()
    await asyncio.sleep(6) 

    record_task = asyncio.ensure_future(mb.record_data(drone))

    print("  [Mission] Starting native waypoint tracking mission...")
    await drone.mission.start_mission()

    start_mission_time = time.time()
    async for progress in drone.mission.mission_progress():
        if progress.current == progress.total:
            print("  [Mission] Waypoint flight completed naturally.")
            break
        # Extend timeout to 90 seconds to give the drone enough time to struggle
        if time.time() - start_mission_time > 90.0:
            print("  [Mission] Reached 90s test limit, forced truncation!")
            break

    mb.is_running = False
    await record_task
    mb.save_data()
    
    print("  [System] Cleaning up and landing...")
    try: await drone.mission.pause_mission()
    except: pass
    await drone.action.land()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--wind', type=str, required=True)
    asyncio.run(run(parser.parse_args()))