import pyperclip

message = "Computer-use agent clipboard test."

pyperclip.copy(message)
copied_text = pyperclip.paste()

print(f"Copied: {copied_text}")
print(f"Success: {copied_text == message}")