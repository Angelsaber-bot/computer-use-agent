from unittest.mock import patch

from computer_agent.control.computer_controller import ComputerController


@patch("computer_agent.control.computer_controller.pyautogui.position")
def test_get_mouse_position(mock_position):
    mock_position.return_value = (100, 200)

    controller = ComputerController()
    result = controller.get_mouse_position()

    assert result == (100, 200)
    mock_position.assert_called_once()