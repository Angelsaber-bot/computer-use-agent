import time
import pyautogui

pyautogui.FAILSAFE = True

print("Move the mouse to a safe blank area.")
time.sleep(5)

target = pyautogui.position()
pyautogui.click(*target)

print(f"Clicked: {target}")