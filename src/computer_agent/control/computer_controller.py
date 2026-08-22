import pyautogui
import pyperclip
import subprocess

class ComputerController:
    def __init__(self):
        pyautogui.FAILSAFE = True

    @staticmethod
    def get_mouse_position():
        return pyautogui.position()

    @staticmethod
    def move_mouse(x, y, duration=0.5):
        pyautogui.moveTo(x, y, duration=duration)

    @staticmethod
    def click_mouse(x, y):
        pyautogui.click(x, y)

    @staticmethod
    def scroll(amount):
        pyautogui.scroll(amount)

    @staticmethod
    def type_text(text, interval=0.05):
        pyautogui.write(text, interval=interval)

    @staticmethod
    def press_key(key):
        pyautogui.press(key)

    @staticmethod
    def hotkey(*keys):
        pyautogui.hotkey(*keys)
    @staticmethod
    def copy_to_clipboard(text):
        pyperclip.copy(text)

    @staticmethod
    def read_from_clipboard():
        return pyperclip.paste()

    @staticmethod
    def paste_text(text):
        pyperclip.copy(text)
        pyautogui.hotkey("command", "v")

    @staticmethod
    def get_screen_size():
        return pyautogui.size()

    @staticmethod
    def capture_screenshot(output_path):
        screenshot = pyautogui.screenshot()
        screenshot.save(output_path)

        return output_path
    
    @staticmethod
    def activate_app(app_name):
        subprocess.run(["open", "-a", app_name], check=True)

    @staticmethod
    def open_url(url, browser="Google Chrome"):
        subprocess.run(["open", "-a", browser, url], check=True)