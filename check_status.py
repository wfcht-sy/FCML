#!/usr/bin/env python3
import asyncio
from mavsdk import System

async def run():
    drone = System()
    print("1. 连接无人机中...")
    await drone.connect(system_address="udpin://0.0.0.0:14540")

    print("2. 等待连接确认...")
    async for state in drone.core.connection_state():
        if state.is_connected: 
            print("   -> [成功] 无人机已连接！")
            break

    # --- 测试位置数据 ---
    print("3. 测试位置数据流 (Position)...")
    try:
        # [修复] 显式获取迭代器的下一个值，并加超时
        # next_pos_task = drone.telemetry.position().__aiter__().__anext__()
        # 更优雅的写法：直接用 anext() (Python 3.10+) 或手动获取
        
        async def get_one_position():
            async for pos in drone.telemetry.position():
                return pos
        
        pos = await asyncio.wait_for(get_one_position(), timeout=3.0)
        print(f"   -> [成功] 收到位置: Alt={pos.relative_altitude_m:.2f}m")
        
    except asyncio.TimeoutError:
        print("   -> [失败] 3秒内未收到位置数据！")

    # --- 测试电机数据 ---
    print("4. 测试电机数据流 (Actuator)...")
    try:
        async def get_one_actuator():
            async for status in drone.telemetry.actuator_output_status():
                return status
                
        status = await asyncio.wait_for(get_one_actuator(), timeout=3.0)
        pwm = status.active[:4]
        
        if len(pwm) == 0:
            print("   -> [警告] 收到数据包但为空 (Len=0)")
        else:
            print(f"   -> [成功] 收到电机数据: {pwm}")
            val = pwm[0]
            if val > 100:
                norm = (val - 1500) / 500
                print(f"      数值分析: 微秒制 (1000-2000) -> 归一化后: {norm:.2f}")
            else:
                norm = val * 2 - 1
                print(f"      数值分析: 比例制 (0-1) -> 归一化后: {norm:.2f}")
            
    except asyncio.TimeoutError:
        print("   -> [失败] 3秒内未收到电机数据！(PX4 未推送)")

if __name__ == "__main__":
    asyncio.run(run())