from unittest.mock import Mock

import pytest

from computer_agent.tools.base import (
    ToolValidationError,
)
from computer_agent.tools.computer import application as application_module
from computer_agent.tools.computer.application import (
    ActivateAppTool,
    OpenURLTool,
)


def test_activate_app_tool():
    controller = Mock()
    tool = ActivateAppTool(controller)

    arguments = tool.validate_arguments(
        {
            "app_name": "TextEdit",
        }
    )

    output = tool.run(**arguments)

    controller.activate_app.assert_called_once_with(
        "TextEdit"
    )

    assert output == {
        "app_name": "TextEdit",
    }


def test_open_url_tool_uses_default_browser():
    controller = Mock()
    tool = OpenURLTool(controller)

    arguments = tool.validate_arguments(
        {
            "url": "https://example.com",
        }
    )

    output = tool.run(**arguments)

    controller.open_url.assert_called_once_with(
        "https://example.com",
        browser="Google Chrome",
    )

    assert output == {
        "url": "https://example.com",
        "browser": "Google Chrome",
    }


def test_open_url_tool_accepts_explicit_safari_browser():
    controller = Mock()
    tool = OpenURLTool(controller)

    arguments = tool.validate_arguments(
        {
            "url": "https://example.com",
            "browser": "Safari",
        }
    )

    output = tool.run(**arguments)

    controller.open_url.assert_called_once_with(
        "https://example.com",
        browser="Safari",
    )
    assert output == {
        "url": "https://example.com",
        "browser": "Safari",
    }


@pytest.mark.parametrize(
    "url",
    [
        "http://example.com",
        "https://example.com",
    ],
)
def test_open_url_tool_accepts_http_and_https_urls(url):
    controller = Mock()
    tool = OpenURLTool(controller)

    arguments = tool.validate_arguments(
        {
            "url": url,
        }
    )

    output = tool.run(**arguments)

    controller.open_url.assert_called_once_with(
        url,
        browser="Google Chrome",
    )
    assert output["url"] == url


def test_open_url_tool_accepts_file_fixture_url():
    controller = Mock()
    fixture_path = (
        application_module._FIXTURE_ROOT
        / "phase03_screen_perception/experiment_11_accessibility_text_input.html"
    )
    fixture_url = fixture_path.resolve().as_uri()
    tool = OpenURLTool(controller)

    arguments = tool.validate_arguments(
        {
            "url": fixture_url,
        }
    )

    output = tool.run(**arguments)

    controller.open_url.assert_called_once_with(
        fixture_url,
        browser="Google Chrome",
    )
    assert output == {
        "url": fixture_url,
        "browser": "Google Chrome",
    }


def test_activate_app_rejects_empty_name():
    controller = Mock()
    tool = ActivateAppTool(controller)

    arguments = tool.validate_arguments(
        {
            "app_name": "   ",
        }
    )

    with pytest.raises(
        ToolValidationError,
        match="argument 'app_name' cannot be empty",
    ):
        tool.run(**arguments)

    controller.activate_app.assert_not_called()


@pytest.mark.parametrize(
    ("arguments", "error_message"),
    [
        (
            {
                "url": "   ",
            },
            "argument 'url' cannot be empty",
        ),
        (
            {
                "url": "https://example.com",
                "browser": "   ",
            },
            "argument 'browser' cannot be empty",
        ),
    ],
)
def test_open_url_rejects_empty_values(
    arguments,
    error_message,
):
    controller = Mock()
    tool = OpenURLTool(controller)

    validated = tool.validate_arguments(arguments)

    with pytest.raises(
        ToolValidationError,
        match=error_message,
    ):
        tool.run(**validated)

    controller.open_url.assert_not_called()


@pytest.mark.parametrize(
    "url",
    [
        "http://",
        "https:///path",
        "https://:443/path",
    ],
)
def test_open_url_rejects_http_or_https_without_host(url):
    controller = Mock()
    tool = OpenURLTool(controller)
    arguments = tool.validate_arguments(
        {
            "url": url,
        }
    )

    with pytest.raises(
        ToolValidationError,
        match="argument 'url' must include a host",
    ):
        tool.run(**arguments)

    controller.open_url.assert_not_called()


def test_open_url_rejects_hostname_containing_whitespace():
    controller = Mock()
    tool = OpenURLTool(controller)
    arguments = tool.validate_arguments(
        {
            "url": "https://exa mple.com",
        }
    )

    with pytest.raises(
        ToolValidationError,
        match=(
            "argument 'url' cannot contain whitespace "
            "or control characters"
        ),
    ):
        tool.run(**arguments)

    controller.open_url.assert_not_called()


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com:99999",
        "https://example.com:abc",
        "https://[::1",
    ],
)
def test_open_url_rejects_invalid_host_or_port(url):
    controller = Mock()
    tool = OpenURLTool(controller)
    arguments = tool.validate_arguments(
        {
            "url": url,
        }
    )

    with pytest.raises(
        ToolValidationError,
        match="argument 'url' must include a valid host and port",
    ):
        tool.run(**arguments)

    controller.open_url.assert_not_called()


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/path\nnext",
        "https://example.com/path\u00a0next",
        "https://example.com/\x1f",
    ],
)
def test_open_url_rejects_whitespace_or_control_characters(url):
    controller = Mock()
    tool = OpenURLTool(controller)
    arguments = tool.validate_arguments(
        {
            "url": url,
        }
    )

    with pytest.raises(
        ToolValidationError,
        match=(
            "argument 'url' cannot contain whitespace "
            "or control characters"
        ),
    ):
        tool.run(**arguments)

    controller.open_url.assert_not_called()


@pytest.mark.parametrize(
    "url",
    [
        "javascript:alert(1)",
        "data:text/plain,hello",
        "ftp://example.com/file",
        "chrome://settings",
    ],
)
def test_open_url_rejects_unsupported_schemes(url):
    controller = Mock()
    tool = OpenURLTool(controller)
    arguments = tool.validate_arguments(
        {
            "url": url,
        }
    )

    with pytest.raises(
        ToolValidationError,
        match="argument 'url' must use http, https, or file scheme",
    ):
        tool.run(**arguments)

    controller.open_url.assert_not_called()


def test_open_url_rejects_file_url_outside_fixture_directory(tmp_path):
    controller = Mock()
    outside_url = (tmp_path / "fixture.html").resolve().as_uri()
    tool = OpenURLTool(controller)
    arguments = tool.validate_arguments(
        {
            "url": outside_url,
        }
    )

    with pytest.raises(
        ToolValidationError,
        match="argument 'url' file path must be inside assets/fixtures",
    ):
        tool.run(**arguments)

    controller.open_url.assert_not_called()


@pytest.mark.parametrize(
    "browser",
    [
        "Firefox",
        "Google Chrome Canary",
        "google chrome",
    ],
)
def test_open_url_rejects_unsupported_browser_names(browser):
    controller = Mock()
    tool = OpenURLTool(controller)
    arguments = tool.validate_arguments(
        {
            "url": "https://example.com",
            "browser": browser,
        }
    )

    with pytest.raises(
        ToolValidationError,
        match="argument 'browser' must be Google Chrome or Safari",
    ):
        tool.run(**arguments)

    controller.open_url.assert_not_called()
