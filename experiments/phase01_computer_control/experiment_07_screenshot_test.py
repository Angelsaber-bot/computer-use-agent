from pathlib import Path
import pyautogui

output = Path(__file__).parents[2] / "assets/screenshots/phase01_computer_control/experiment_07_screen.png"

screenshot = pyautogui.screenshot()
screenshot.save(output)

print(f"Screenshot saved: {output}")