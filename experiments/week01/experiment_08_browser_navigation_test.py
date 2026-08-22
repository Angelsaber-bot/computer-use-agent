import subprocess

url = "https://example.com"

subprocess.run(["open", "-a", "Google Chrome", url], check=True)

print(f"Opened: {url}")