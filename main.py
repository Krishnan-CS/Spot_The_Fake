from launch_URLs import launch_URLs, get_url_list, stop_server
from utils import capture_html_screenshot, save_html_content
from ui_similarity import check_UI_Similarity
from transformers import CLIPProcessor, CLIPModel
import time, os
from ollama_model import OllamaGemmaModel

LOCAL_HOST_PATH = "http://localhost:8000"


# ------------- Main code -----------------------

# Launch URLs
url_list = get_url_list(1)  # Specify number of urls to open
launch_URLs(url_list)

# Capture HTML content
file_paths = [os.getcwd() + path for path in url_list]
save_html_content(file_paths)

# Capture screenshot of URLs
# This can capture browser content perfectly but runs slow.
# url_paths = [LOCAL_HOST_PATH + path for path in url_list]
# capture_html_screenshot(url_paths)


# Load Gemma model
OLLAMA_API_BASE_URL = "http://localhost:11434/api/generate" # Default Ollama API endpoint
OLLAMA_MODEL_NAME = "gemma3:1b" # Or "gemma:7b" if you pulled that one
gemma_model = OllamaGemmaModel(OLLAMA_MODEL_NAME, OLLAMA_API_BASE_URL)

# LLM analysis from text
llm_result = {}
html_text_folder = os.getcwd() + '/HTML_Text/'
html_text_files = [f for f in os.listdir(html_text_folder)]
for file in html_text_files:
    llm_result[file.split('.')[0]] = gemma_model.analyze_html_with_gemma(html_text_folder + file)

print(llm_result)

# Load CLIP once
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# UI Similarity from screenshots
similarity_result = {}
screenshots_folder = os.getcwd() + '/Screenshots/'
screenshot_files = [f for f in os.listdir(screenshots_folder)]
for screenshot in screenshot_files:
    similarity_result[screenshot.split('.')[0]] = check_UI_Similarity(screenshots_folder + screenshot, clip_model, clip_processor)

print(similarity_result)


# Integrate Vishnu's whois meta data

# Open dashboard to show the results


# Stop the server after running all the code
stop_server()
print('Server stopped!')