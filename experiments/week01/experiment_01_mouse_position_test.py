import time

import pyautogui


print("Move the mouse to any location.")
time.sleep(3)

position = pyautogui.position()
print(f"Mouse position: x={position.x}, y={position.y}")