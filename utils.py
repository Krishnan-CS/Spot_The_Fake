from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import os, re


def capture_html_screenshot(url_paths, viewport=(1280, 800), retries=3):
    for url in url_paths:
        for attempt in range(retries):
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page(viewport={"width": viewport[0], "height": viewport[1]})
                print(f'Capturing screenshot from {url} (Attempt {attempt+1})')
                output_path = f"Screenshots/{url.split('/')[-1]}.png"
                try:
                    page.goto(url, wait_until="networkidle", timeout=60000)
                    page.screenshot(path=output_path, full_page=False)
                    print(f"Screenshot saved to {output_path}")
                    break  # Success, exit retry loop
                except Exception as e:
                    print(f'Timeout—capturing partial render: {e}')
                finally:
                    browser.close()


def clean_html_for_llm(html_content):
    """
    Cleans HTML content by extracting visible text and removing excessive whitespace,
    making it more digestible for the LLM.
    """
    try:
        soup = BeautifulSoup(html_content, 'html.parser')

        # Remove script and style tags as they often contain code noise
        for script_or_style in soup(["script", "style"]):
            script_or_style.extract()

        # Extract visible text
        text = soup.get_text()

        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for phrase in lines if phrase.strip())
        cleaned_text = ' '.join(chunks)

        # Further reduce multiple spaces to a single space
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

        # Limit the text length to avoid exceeding LLM input limits
        MAX_TEXT_LENGTH = 15000
        if len(cleaned_text) > MAX_TEXT_LENGTH:
            return cleaned_text[:MAX_TEXT_LENGTH] + "..."
        return cleaned_text
    except Exception as e:
        print(f"Error cleaning HTML: {e}")
        return ""

def save_html_content(file_paths):
    os.makedirs("HTML_Text", exist_ok=True)   # Ensure the output directory exists
    for file in file_paths:
        if file and os.path.exists(file):
            with open(file, 'r', encoding='utf-8') as f:
                html_content = f.read()

                text_from_html = clean_html_for_llm(html_content)
                output_file_path = f"HTML_Text/{file.split('/')[-1]}.txt"
                with open(output_file_path, 'w', encoding='utf-8') as out_f:
                    out_f.write(text_from_html)

                print(f'HTML Text saved in {output_file_path}')