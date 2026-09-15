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
    def release_modifier_keys(keys=None):
        if keys is None:
            keys = (
                "command",
                "ctrl",
                "alt",
                "option",
                "shift",
                "fn",
            )

        for key in keys:
            pyautogui.keyUp(key)

    @staticmethod
    def press_key(key):
        pyautogui.press(key)

    @staticmethod
    def hotkey(*keys, interval=0.1):
        pyautogui.hotkey(*keys, interval=interval)
    @staticmethod
    def copy_to_clipboard(text):
        pyperclip.copy(text)

    @staticmethod
    def read_from_clipboard():
        return pyperclip.paste()

    @staticmethod
    def paste_text(text):
        pyperclip.copy(text)
        ComputerController.hotkey("command", "v")

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
    def open_task_chrome_window(
        url,
        task_marker,
    ):
        """Create a Chrome window with a persistent task marker tab."""
        if (
            not isinstance(url, str)
            or not url.strip()
        ):
            raise ValueError(
                "url must be a non-empty string"
            )

        if (
            not isinstance(task_marker, str)
            or not task_marker.strip()
        ):
            raise ValueError(
                "task_marker must be a non-empty string"
            )

        marker_url = (
            "about:blank#computer-agent-task="
            + task_marker.strip()
        )

        script = """
on run argv
    set targetURL to item 1 of argv
    set markerURL to item 2 of argv

    tell application "Google Chrome"
        set taskWindow to make new window

        set URL of active tab of taskWindow to targetURL

        make new tab at end of tabs of taskWindow ¬
            with properties {URL:markerURL}

        set active tab index of taskWindow to 1
        set index of taskWindow to 1
        activate
    end tell
end run
"""

        subprocess.run(
            [
                "osascript",
                "-e",
                script,
                url,
                marker_url,
            ],
            check=True,
        )

        return marker_url

    @staticmethod
    def activate_task_chrome_window(
        marker_url,
        working_url_prefix,
    ):
        """Activate the marker-owned Chrome window and working tab."""
        if (
            not isinstance(marker_url, str)
            or not marker_url.startswith(
                "about:blank#computer-agent-task="
            )
        ):
            raise ValueError(
                "marker_url must be a valid "
                "Computer Agent Chrome marker"
            )

        if (
            not isinstance(working_url_prefix, str)
            or not working_url_prefix.strip()
        ):
            raise ValueError(
                "working_url_prefix must be a non-empty string"
            )

        script = """
on run argv
    set markerURL to item 1 of argv
    set workingURLPrefix to item 2 of argv
    set foundWindowCount to 0
    set selectedWindow to missing value
    set selectedTabIndex to 0

    tell application "Google Chrome"
        repeat with candidateWindow in windows
            set markerFoundInWindow to false
            set workingTabCount to 0
            set workingTabIndex to 0
            set tabIndex to 0

            repeat with candidateTab in tabs of candidateWindow
                set tabIndex to tabIndex + 1

                if URL of candidateTab is markerURL then
                    set markerFoundInWindow to true
                else if URL of candidateTab starts with workingURLPrefix then
                    set workingTabCount to workingTabCount + 1
                    set workingTabIndex to tabIndex
                end if
            end repeat

            if markerFoundInWindow then
                set foundWindowCount to foundWindowCount + 1

                if workingTabCount is not 1 then
                    error "Agent Chrome task working tab is missing or ambiguous"
                end if

                set selectedWindow to candidateWindow
                set selectedTabIndex to workingTabIndex
            end if
        end repeat

        if foundWindowCount is 0 then
            error "Agent Chrome task window not found"
        end if

        if foundWindowCount is greater than 1 then
            error "Agent Chrome task marker is ambiguous"
        end if

        set minimized of selectedWindow to false
        set index of selectedWindow to 1
        set active tab index of selectedWindow to selectedTabIndex
        activate
        return "found"
    end tell
end run
"""

        try:
            subprocess.run(
                [
                    "osascript",
                    "-e",
                    script,
                    marker_url,
                    working_url_prefix,
                ],
                check=True,
                stdout=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError as error:
            raise RuntimeError(
                "The Agent-owned Chrome task window "
                "could not be safely resumed."
            ) from error

    @staticmethod
    def open_url(
        url,
        browser="Google Chrome",
        new_window=False,
    ):
        if not new_window:
            subprocess.run(
                [
                    "open",
                    "-a",
                    browser,
                    url,
                ],
                check=True,
            )
            return

        if browser != "Google Chrome":
            raise ValueError(
                "new_window is currently supported "
                "only for Google Chrome"
            )

        script = """
on run argv
    set targetURL to item 1 of argv

    tell application "Google Chrome"
        set taskWindow to make new window
        set URL of active tab of taskWindow to targetURL
        activate
    end tell
end run
"""

        subprocess.run(
            [
                "osascript",
                "-e",
                script,
                url,
            ],
            check=True,
        )
