#!/usr/bin/env python3
"""
e-puck Left-Hand Maze Solver - Final Version with PD Controller & Odometry
"""
import math
from enum import Enum
import numpy as np
from controller import Robot

# ---------------- ODOMETRY FUNCTIONS ----------------
def get_wheels_speed(encoderValues, oldEncoderValues, pulses_per_turn, delta_t):
    ang_diff_l = 2 * np.pi * (encoderValues[0] - oldEncoderValues[0]) / pulses_per_turn
    ang_diff_r = 2 * np.pi * (encoderValues[1] - oldEncoderValues[1]) / pulses_per_turn
    return ang_diff_l / delta_t, ang_diff_r / delta_t

def get_robot_speeds(wl, wr, R, D):
    u = R / 2.0 * (wr + wl)
    w = R / D * (wr - wl)
    return u, w

def get_robot_pose(u, w, x_old, y_old, phi_old, delta_t):
    delta_phi = w * delta_t
    phi = phi_old + delta_phi
    phi = (phi + np.pi) % (2 * np.pi) - np.pi  # Normalize between [-π, π]
    delta_x = u * np.cos(phi) * delta_t
    delta_y = u * np.sin(phi) * delta_t
    return x_old + delta_x, y_old + delta_y, phi

# ---------------- MAIN CONTROL ----------------
def main():
    # Robot & Time Setup
    robot = Robot()
    timestep = int(robot.getBasicTimeStep())

    # Odometry Setup
    pulses_per_turn = 1000
    R = 0.0205
    D = 0.052
    x, y, phi = 0.0, 0.0, 0.0

    # Device Setup
    left_motor = robot.getDevice('left wheel motor')
    right_motor = robot.getDevice('right wheel motor')
    left_encoder = robot.getDevice('left wheel sensor')
    right_encoder = robot.getDevice('right wheel sensor')
    tof = robot.getDevice("tof")
    ir_left = robot.getDevice("ps5")
    ir_front_left = robot.getDevice("ps6")
    ir_right = robot.getDevice("ps2")
    ir_front_right = robot.getDevice("ps1")

    for d in [left_encoder, right_encoder, tof, ir_left, ir_front_left, ir_right, ir_front_right]:
        d.enable(timestep)

    left_motor.setPosition(float('inf'))
    right_motor.setPosition(float('inf'))
    left_motor.setVelocity(0)
    right_motor.setVelocity(0)

    # Constants
    MAX_SPEED = 6.28
    BASE_SPEED = 0.9 * MAX_SPEED
    TURN_SPEED = 0.6 * MAX_SPEED
    IR_TARGET_DIST = 350
    IR_LEFT_WALL_TH = 150
    IR_RIGHT_WALL_TH = 150
    TOF_FRONT_TH = 100.0
    WALL_FOLLOW_KP = 0.007
    WALL_FOLLOW_KD = 0.08
    STRAIGHT_DURATION = 0.1
    RADS_FOR_90_DEG = (D * math.pi / 4) / R
    RADS_FOR_180_DEG = (D * math.pi / 2) / R

    # State Machine
    class State(Enum):
        FOLLOW_WALL = 1
        LEFT_TURN = 2
        RIGHT_TURN = 3
        U_TURN = 4
        DRIVE_STRAIGHT = 5

    state = State.FOLLOW_WALL
    start_time = 0.0
    start_pos_left = 0.0
    start_pos_right = 0.0
    last_error = 0.0

    robot.step(timestep)
    oldEncoderValues = [left_encoder.getValue(), right_encoder.getValue()]

    print("🚀 Starting Maze Solver + Odometry...\n")

    while robot.step(timestep) != -1:
        time_now = robot.getTime()

        # Sensor Readings
        tof_val = tof.getValue()
        left_val = max(ir_left.getValue(), ir_front_left.getValue())
        right_val = max(ir_right.getValue(), ir_front_right.getValue())
        pos_left = left_encoder.getValue()
        pos_right = right_encoder.getValue()
        encoderValues = [pos_left, pos_right]
        delta_t = timestep / 1000.0

        # Odometry
        wl, wr = get_wheels_speed(encoderValues, oldEncoderValues, pulses_per_turn, delta_t)
        u, w = get_robot_speeds(wl, wr, R, D)
        x, y, phi = get_robot_pose(u, w, x, y, phi, delta_t)
        oldEncoderValues = encoderValues.copy()

        # --- State Machine ---
        def set_speeds(vl, vr):
            left_motor.setVelocity(max(-MAX_SPEED, min(vl, MAX_SPEED)))
            right_motor.setVelocity(max(-MAX_SPEED, min(vr, MAX_SPEED)))

        if state == State.FOLLOW_WALL:
            if tof_val < TOF_FRONT_TH:
                set_speeds(0, 0)
                has_left = left_val > IR_LEFT_WALL_TH
                has_right = right_val > IR_RIGHT_WALL_TH
                if has_left and has_right:
                    state = State.U_TURN
                    start_pos_left = pos_left
                elif has_left:
                    state = State.RIGHT_TURN
                    start_pos_left = pos_left
                else:
                    state = State.LEFT_TURN
                    start_pos_right = pos_right
            else:
                error = left_val - IR_TARGET_DIST
                derivative = error - last_error
                correction = WALL_FOLLOW_KP * error + WALL_FOLLOW_KD * derivative
                set_speeds(BASE_SPEED - correction, BASE_SPEED + correction)
                last_error = error

        elif state == State.LEFT_TURN:
            set_speeds(-TURN_SPEED, TURN_SPEED)
            if abs(pos_right - start_pos_right) >= RADS_FOR_90_DEG:
                state = State.DRIVE_STRAIGHT
                start_time = time_now

        elif state == State.RIGHT_TURN:
            set_speeds(TURN_SPEED, -TURN_SPEED)
            if abs(pos_left - start_pos_left) >= RADS_FOR_90_DEG:
                state = State.DRIVE_STRAIGHT
                start_time = time_now

        elif state == State.U_TURN:
            set_speeds(TURN_SPEED, -TURN_SPEED)
            if abs(pos_left - start_pos_left) >= RADS_FOR_180_DEG:
                state = State.DRIVE_STRAIGHT
                start_time = time_now

        elif state == State.DRIVE_STRAIGHT:
            set_speeds(BASE_SPEED, BASE_SPEED)
            if time_now - start_time > STRAIGHT_DURATION:
                state = State.FOLLOW_WALL
                last_error = 0

        # Print odometry result
        print(f"[{state.name}] x={x:.3f} m, y={y:.3f} m, θ={math.degrees(phi):.1f}°")

if __name__ == '__main__':
    main()
