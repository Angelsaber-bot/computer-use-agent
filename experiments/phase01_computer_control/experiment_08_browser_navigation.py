import subprocess


def main() -> None:
    url = "https://example.com"

    subprocess.run(["open", "-a", "Google Chrome", url], check=True)

    print(f"Opened: {url}")


if __name__ == "__main__":
    main()
