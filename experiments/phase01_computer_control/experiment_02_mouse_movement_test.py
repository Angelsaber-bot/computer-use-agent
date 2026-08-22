import time
import pyautogui

pyautogui.FAILSAFE = True

start = pyautogui.position()
screen = pyautogui.size()
center = (screen.width // 2, screen.height // 2)

print(f"Start: {start}")
print(f"Target: {center}")

time.sleep(3)
pyautogui.moveTo(*center, duration=1)
print(f"Current: {pyautogui.position()}")

pyautogui.moveTo(*start, duration=1)
print(f"Final: {pyautogui.position()}")