import pyautogui


class ComputerController:
    def __init__(self):
        pyautogui.FAILSAFE = True

    @staticmethod
    def get_mouse_position():
        return pyautogui.position()