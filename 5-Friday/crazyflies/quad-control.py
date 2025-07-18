import numpy as np
import logging
import time
import sys
import copy

from cflib.crtp import init_drivers
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.position_hl_commander import PositionHlCommander
from cflib.crazyflie.high_level_commander import HighLevelCommander
from cflib.crazyflie.log import LogConfig

global_x = 0.
global_y = 0.
global_z = 0.
global_roll = 0.
global_pitch = 0.
global_yaw = 0.

def make_uri(radio_id, channel, datarate, address_hex):
    return f"radio://{radio_id}/{channel}/{datarate}/{address_hex}"

def check_connection_leds_signal(scf):
    try:
        scf.cf.param.set_value("led.bitmask", 255)
        time.sleep(2)
        scf.cf.param.set_value("led.bitmask", 0)
        time.sleep(1)
    except Exception as e:
        print(f"Error during connection leds signal!: {e}")

def check_lighthouse_deck(scf):
    lhdeck_value = scf.cf.param.get_value(complete_name = "deck.bcLighthouse4", timeout = 1)
    if int(lhdeck_value):
        print(f"[{uri}] -> Lighthouse deck attached!")
    else:
        print(f"[{uri}] -> Lighthouse deck NOT attached!")
        sys.exit(1)

def log_state_callback(uri):
    def callback(timestamp, data, logconf):
        global global_x, global_y, global_z, global_roll, global_pitch, global_yaw
        global_x, global_y, global_z, global_roll, global_pitch, global_yaw = data['stateEstimate.x'], data['stateEstimate.y'], data['stateEstimate.z'], data['stateEstimate.roll'], data['stateEstimate.pitch'], data['stateEstimate.yaw']
        print(f"[t = {timestamp}]: [{uri}] ->", \
                f"\n\t-> x = {data['stateEstimate.x']:.3f}, y = {data['stateEstimate.y']:.3f}, z = {data['stateEstimate.z']:.3f}", \
                f"\n\t-> roll = {data['stateEstimate.roll']:.3f}, pitch = {data['stateEstimate.pitch']:.3f}, yaw = {data['stateEstimate.yaw']:.3f}")
    return callback

def log_battery_callback(uri):
    def callback(timestamp, data, logconf):
        print(f" - [t = {timestamp}]: [{uri}] -> {logconf.name}", \
                f"\n\t-> Battery level = {data['pm.batteryLevel']:3d} %")
    return callback

def log_motors_callback(uri):
    def callback(timestamp, data, logconf):
        print(f" - [t = {timestamp}]: [{uri}] -> {logconf.name}", \
                f"\n\t-> m1 = {data['motor.m1']}, m2 = {data['motor.m2']}, m3 = {data['motor.m3']}, m4 = {data['motor.m4']}")
    return callback

def create_target_position():
    xd, yd, zd, yawd = 0.0, 0.0, 0.0, 0.0
    fly_target = True
    while True:
        try:
            xd = float(input("x desired (m): ").strip())
            yd = float(input("y desired (m): ").strip())
            zd = float(input("z desired (m): ").strip())
            yawd = float(input("yaw desired (degrees): ").strip())
        except ValueError:
            print("❌ Invalid number entered. Try again:")
            continue
        confirm_target = input("Press ENTER to proceed with the target, or type 'q' and press ENTER to land: ")
        if confirm_target == "":
            print("✅ Confirmed, proceeding to fly!")
            break
        elif confirm_target == "q":
            print("⚠️ Landing now!")
            fly_target = False
            break
    return xd, yd, zd, yawd, fly_target


if __name__ == "__main__":
    logging.basicConfig(level = logging.ERROR)
    init_drivers()
    default_height = 0.5  # default height at takeoff in meters

    uri = make_uri(0, 80, "2M", "E7E7E7E701") # you need to put your drone's URI (change E7E7E7E701)
    try:
        scf = SyncCrazyflie(uri, cf = Crazyflie(rw_cache = "./cache"))
        scf.open_link()
        print(f"[{uri}] -> Connected!")
        check_connection_leds_signal(scf)
        check_lighthouse_deck(scf)
    except Exception as e:
        print(f"Failed to connect to {uri}!: {e}")
        sys.exit(1)

    # We set the controller to our custom controller!!
    # 6: our custom controller
    # 1: Crazyflie's PID controller
    scf.cf.param.set_value("stabilizer.controller", 6)
    time.sleep(2)
    print(scf.cf.param.get_value("stabilizer.controller", timeout = 5))

    hlc = HighLevelCommander(scf.cf)
    time.sleep(1)

    ### Logging variables
    log_state = LogConfig(name = "Current State", period_in_ms = 100)
    log_state.add_variable("stateEstimate.x", "float")
    log_state.add_variable("stateEstimate.y", "float")
    log_state.add_variable("stateEstimate.z", "float")
    log_state.add_variable("stateEstimate.roll", "float")
    log_state.add_variable("stateEstimate.pitch", "float")
    log_state.add_variable("stateEstimate.yaw", "float")
    scf.cf.log.add_config(log_state)
    log_state.data_received_cb.add_callback(log_state_callback(uri))

    scf.cf.console.receivedChar.add_callback(lambda text: print(text))

    log_state.start()
    ### End of Logging

    # Waiting in order to take a valid x, y, z position
    time.sleep(2.)
    x_init, y_init = copy.copy(global_x), copy.copy(global_y)

    # Control!
    takeoff_time = 2.0
    land_time = 3.0
    goto_time = 10.0
    try:
        hlc.takeoff(absolute_height_m = default_height, duration_s = takeoff_time)
        time.sleep(takeoff_time)
        while True:
            time.sleep(1)
            # xd, yd, zd, yawd, fly_target = create_target_position()
            ### TO-DO: Once your LQR works okaysih. You can try different targets! Be carefull of the other teams/drones!
            xd, yd, zd, yawd, fly_target = x_init, y_init, 1.0, 0.0, True # fly one meter above your initial state!
            if fly_target:
                hlc.go_to(x = xd, y = yd, z = zd, yaw = np.deg2rad(yawd), duration_s = goto_time)
                time.sleep(goto_time)
            else:
                hlc.land(absolute_height_m = 0.0, duration_s = land_time)
                time.sleep(land_time)
                break
    finally:
        try:
            hlc.land(absolute_height_m = 0.0, duration_s = land_time)
            time.sleep(land_time)
        except:
            pass

    hlc.land(absolute_height_m = 0.0, duration_s = land_time)

    log_state.stop()
    scf.close_link()
