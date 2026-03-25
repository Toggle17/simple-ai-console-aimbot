import cv2
import numpy as np
import dxcam
from ultralytics import YOLO
import pygetwindow as gw
import math
import vgamepad as vg
import time
import threading


# CONTROLLER SETUP


gamepad = vg.VDS4Gamepad()


# LOAD MODEL


model = YOLO("best2.pt")

try:
    model.to("cuda")
    print("Running YOLO on GPU")
except:
    print("Running YOLO on CPU")

# FIND XBOX WINDOW

windows = gw.getWindowsWithTitle("Xbox")

if len(windows) == 0:
    print("Xbox App window not found!")
    exit()

xbox = windows[0]

monitor = {
    "top": xbox.top,
    "left": xbox.left,
    "width": xbox.width,
    "height": xbox.height
}


# CAMERA


camera = dxcam.create()
camera.start(target_fps=120)


# PID VARIABLES


kp = 0.9
ki = 0.02
kd = 0.35

integral_x = 0
integral_y = 0

prev_error_x = 0
prev_error_y = 0

MAX_INTEGRAL = 5000

# PARAMETERS


MAX_STICK = 127
SENSITIVITY = 1.0

CENTER_THRESHOLD = 5
SHOOT_THRESHOLD = 12

FOV_RADIUS = 220
AIM_BOX_SIZE = 300

TARGET_LOCK_DISTANCE = 120
SHOOT_DELAY = 0.02

last_shot_time = 0


# TARGET LOCK


locked_target = None


# JOYSTICK STATE


joystick_state = {"x":128,"y":128}
state_lock = threading.Lock()


# JOYSTICK THREAD


def joystick_thread():

    while True:

        with state_lock:
            x = joystick_state["x"]
            y = joystick_state["y"]

        gamepad.right_joystick(x_value=x,y_value=y)
        gamepad.update()

        time.sleep(0.004)

thread = threading.Thread(target=joystick_thread, daemon=True)
thread.start()


# FPS


prev_time = time.time()


# MAIN LOOP


while True:

    frame = camera.get_latest_frame()

    if frame is None:
        continue

    top = max(0, monitor["top"])
    left = max(0, monitor["left"])
    bottom = top + monitor["height"]
    right = left + monitor["width"]

    frame = frame[top:bottom,left:right]

    if frame.size == 0:
        continue

    frame = frame.copy()

    screen_center_x = monitor["width"] // 2
    screen_center_y = monitor["height"] // 2

    
    # AIM BOX
    

    half = AIM_BOX_SIZE // 2

    box_left = max(0, screen_center_x - half)
    box_top = max(0, screen_center_y - half)

    box_right = min(monitor["width"], screen_center_x + half)
    box_bottom = min(monitor["height"], screen_center_y + half)

    aim_region = frame[box_top:box_bottom, box_left:box_right]

    scale = 0.6
    small = cv2.resize(aim_region,(0,0),fx=scale,fy=scale)

    results = model(small, verbose=False)

    targets = []

    for result in results:
        for box in result.boxes:

            x1,y1,x2,y2 = map(int,box.xyxy[0])

            x1 = int(x1/scale) + box_left
            y1 = int(y1/scale) + box_top
            x2 = int(x2/scale) + box_left
            y2 = int(y2/scale) + box_top

            conf = float(box.conf)

            if conf > 0.5:

                cx = int((x1+x2)/2)
                cy = int((y1+y2)/2)

                dist = math.hypot(cx-screen_center_x, cy-screen_center_y)

                if dist < FOV_RADIUS:
                    targets.append((cx,cy,x1,y1,x2,y2,dist))

    chosen = None


    if targets:

        chosen = min(targets, key=lambda t: t[6])

    if chosen:

        cx,cy,x1,y1,x2,y2,_ = chosen

        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.circle(frame,(cx,cy),5,(0,0,255),-1)

        error_x = cx - screen_center_x
        error_y = cy - screen_center_y

        # AUTOSHOOT

        if abs(error_x) < SHOOT_THRESHOLD and abs(error_y) < SHOOT_THRESHOLD:

            if time.time() - last_shot_time > SHOOT_DELAY:

                gamepad.right_trigger(value=255)
                gamepad.update()

                time.sleep(0.02)

                gamepad.right_trigger(value=0)
                gamepad.update()

                last_shot_time = time.time()

        # AIM PID

        integral_x += error_x
        integral_y += error_y

        integral_x = max(min(integral_x,MAX_INTEGRAL),-MAX_INTEGRAL)
        integral_y = max(min(integral_y,MAX_INTEGRAL),-MAX_INTEGRAL)

        derivative_x = error_x - prev_error_x
        derivative_y = error_y - prev_error_y

        prev_error_x = error_x
        prev_error_y = error_y

        output_x = (kp*error_x) + (ki*integral_x) + (kd*derivative_x)
        output_y = (kp*error_y) + (ki*integral_y) + (kd*derivative_y)

        distance = math.hypot(output_x, output_y)

        if distance > 0:

            angle = math.atan2(output_y, output_x)

            magnitude = min(distance * SENSITIVITY, MAX_STICK)

            stick_x = int(math.cos(angle) * magnitude)
            stick_y = int(math.sin(angle) * magnitude)

        else:
            stick_x = 0
            stick_y = 0

        stick_x = int(np.clip(128 + stick_x,0,255))
        stick_y = int(np.clip(128 + stick_y,0,255))

        with state_lock:
            joystick_state["x"]=stick_x
            joystick_state["y"]=stick_y

    else:

        with state_lock:
            joystick_state["x"]=128
            joystick_state["y"]=128

    # UI

    cv2.circle(frame,(screen_center_x,screen_center_y),4,(255,255,0),-1)
    cv2.circle(frame,(screen_center_x,screen_center_y),FOV_RADIUS,(255,0,255),2)

    cv2.rectangle(frame,(box_left,box_top),(box_right,box_bottom),(0,255,255),1)

    new_time=time.time()
    fps=int(1/(new_time-prev_time))
    prev_time=new_time

    cv2.putText(frame,f"FPS: {fps}",(20,40),
    cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2)

    cv2.imshow("AI Target Tracker",frame)

    if cv2.waitKey(1)==27:
        break

cv2.destroyAllWindows()
camera.stop()
