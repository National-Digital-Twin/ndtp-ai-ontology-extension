import sys
import datetime
import inspect


def log(message, level="INFO"):
    """
    Log a message to the terminal with timestamp and source information.

    Args:
        message (str): The message to log
        level (str): Log level (INFO, WARNING, ERROR, DEBUG)
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Get the calling frame information
    caller_frame = inspect.currentframe().f_back
    caller_module = inspect.getmodule(caller_frame)
    module_name = caller_module.__name__ if caller_module else "unknown"

    # Format the log message
    formatted_message = f"[{timestamp}] [{level}] [{module_name}] {message}"

    # Print to terminal
    print(formatted_message, file=sys.stderr)
