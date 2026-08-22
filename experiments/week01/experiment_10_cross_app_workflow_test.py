from pathlib import Path
import subprocess
import time
import pyautogui
import pyperclip

url = "https://example.com"
output = Path(__file__).parents[2] / "assets/screenshots/experiment_10_workflow.png"

pyperclip.copy(url)
subprocess.run(["open", "-a", "Google Chrome", url], check=True)
time.sleep(2)

script = 'tell application "TextEdit" to activate\ntell application "TextEdit" to make new document'
subprocess.run(["osascript", "-e", script], check=True)
time.sleep(1)

pyautogui.write(pyperclip.paste(), interval=0.05)
pyautogui.screenshot().save(output)

print("Cross-app workflow completed.")