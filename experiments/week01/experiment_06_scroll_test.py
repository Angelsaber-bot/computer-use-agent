import time
import pyautogui

pyautogui.FAILSAFE = True

print("Switch to a scrollable page.")
time.sleep(5)

pyautogui.scroll(-5)
print("Scrolled down.")

time.sleep(2)

pyautogui.scroll(5)
print("Scrolled up.")