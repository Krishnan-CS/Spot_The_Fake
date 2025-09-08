import os
import random
from http.server import BaseHTTPRequestHandler, HTTPServer
import threading
import webbrowser
import time
import pyautogui

# Define a custom HTTP request handler that takes html_content
class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        # Serve the HTML file based on the requested path
        url_path = self.path
        file_path = url_map.get(url_path)
        if file_path and os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(html_content.encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'404 Not Found')

# Store the mapping of URL paths to HTML file paths
url_map = {}
httpd_instance = None

# Function to select a random HTML file
def get_url_path():
    folder_path_genuine = '/Test_HTML/Genuine/'
    folder_path_phishing = '/Test_HTML/Phishing/'
    folder_path = random.choice([folder_path_genuine, folder_path_phishing])

    full_path = os.getcwd() + folder_path
    files = [f for f in os.listdir(full_path) if f.endswith('.html')]
    if not files:
        return None

    return folder_path + random.choice(files)

def get_url_list(list_size=5):
    url_list = []
    for _ in range(list_size):
        url = get_url_path()
        url_list.append(url)

    return url_list

# Set up and start the HTTP server in a separate thread
def run_server():
    global httpd_instance
    server_address = ('localhost', 8000)
    httpd_instance = HTTPServer(server_address, SimpleHTTPRequestHandler)
    print(f"Serving HTML content on http://{server_address[0]}:{server_address[1]}")
    httpd_instance.serve_forever()

def stop_server():
    global httpd_instance
    if httpd_instance:
        print("Stopping server...")
        httpd_instance.shutdown()
        httpd_instance.server_close()
        httpd_instance = None

# Launch URL
def launch_URLs(url_list, delay=5):
    # Build the url_map: map URL paths to file system paths
    for url_path in url_list:
        os_file_path = os.getcwd() + url_path
        url_map[url_path] = os_file_path

    # Start the server once
    server_thread = threading.Thread(target=run_server)
    server_thread.daemon = True
    server_thread.start()

    # Open each URL in the browser
    for url_path in url_list:
        url = "http://localhost:8000" + url_path
        print(f"Opening {url} in web browser...")
        webbrowser.open(url)
        time.sleep(delay)

        # Capture screenshot of the whole screen
        screenshot_path = f"/phishing-web/public/Screenshots/{url_path.split('/')[-1]}.png"
        screenshot =pyautogui.screenshot(region=(0, 150, 2000, 1200))
        screenshot.save(screenshot_path)
        print(f"Screenshot saved to {screenshot_path}")

def is_alive(url):
    try:
        return requests.get(url).status_code == 200
    except:
        return False