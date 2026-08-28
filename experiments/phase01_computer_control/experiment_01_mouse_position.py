import time

import pyautogui


def main() -> None:
    print("Move the mouse to any location.")
    time.sleep(3)

    position = pyautogui.position()
    print(f"Mouse position: x={position.x}, y={position.y}")


if __name__ == "__main__":
    main()
