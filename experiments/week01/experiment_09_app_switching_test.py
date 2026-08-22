import subprocess
import time

apps = ["TextEdit", "Google Chrome"]

for app in apps:
    subprocess.run(
        ["osascript", "-e", f'tell application "{app}" to activate'],
        check=True
    )
    print(f"Activated: {app}")
    time.sleep(2)