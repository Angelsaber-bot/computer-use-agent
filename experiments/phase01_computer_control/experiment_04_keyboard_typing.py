import subprocess
import time
import pyautogui


def main() -> None:
    pyautogui.FAILSAFE = True

    script = 'tell application "TextEdit" to activate\ntell application "TextEdit" to make new document'
    subprocess.run(["osascript", "-e", script])

    time.sleep(1)
    message = "Computer-use agent keyboard test."
    pyautogui.write(message, interval=0.05)

    print(f"Typed: {message}")


if __name__ == "__main__":
    main()
