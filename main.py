from launch_URLs import launch_URLs, get_url_list, stop_server
from utils import capture_html_screenshot, save_html_content
import time, os

LOCAL_HOST_PATH = "http://localhost:8000"


# ------------- Main code -----------------------

# Launch URLs
url_list = get_url_list(5)  # Specify number of urls to open
launch_URLs(url_list)

# Capture HTML content
file_paths = [os.getcwd() + path for path in url_list]
save_html_content(file_paths)

# Capture screenshot of URLs
# This can capture browser content perfectly but runs slow.
# url_paths = [LOCAL_HOST_PATH + path for path in url_list]
# capture_html_screenshot(url_paths)

# LLM analysis from text

# UI Similarity from screenshots


# Open dashboard to show the results


# Stop the server after running all the code
stop_server()
print('Server stopped!')