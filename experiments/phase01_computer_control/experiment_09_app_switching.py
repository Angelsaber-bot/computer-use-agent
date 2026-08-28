import subprocess
import time


def main() -> None:
    apps = ["TextEdit", "Google Chrome"]

    for app in apps:
        subprocess.run(
            ["osascript", "-e", f'tell application "{app}" to activate'],
            check=True
        )
        print(f"Activated: {app}")
        time.sleep(2)


if __name__ == "__main__":
    main()
