import time
import pyautogui


def main() -> None:
    pyautogui.FAILSAFE = True

    print("Switch to a scrollable page.")
    time.sleep(5)

    pyautogui.scroll(-5)
    print("Scrolled down.")

    time.sleep(2)

    pyautogui.scroll(5)
    print("Scrolled up.")


if __name__ == "__main__":
    main()
