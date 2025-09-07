import os
import pandas as pd
from bs4 import BeautifulSoup
import re
import json
import requests # Import the requests library for API calls

# --- Ollama Configuration ---
OLLAMA_API_BASE_URL = "http://localhost:11434/api/generate" # Default Ollama API endpoint
OLLAMA_MODEL_NAME = "llama3.2:latest" # Or "gemma:7b" if you pulled that one

BASE_DIR = "/html_data/training/"

OUTPUT_CSV_FILE = "phishing_dataset.csv"

FILE_LIMIT_PER_CATEGORY = 1000

# --- Ollama Model Interaction Class ---
class OllamaGemmaModel:
    def __init__(self, model_name, api_url):
        self.model_name = model_name
        self.api_url = api_url
        print(f"Initializing Ollama client for model: {self.model_name} at {self.api_url}")

    def generate_content(self, prompt):
        headers = {"Content-Type": "application/json"}
        data = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False, # We want the full response at once
            "options": {
                "temperature": 0.1, # Keep temperature low for consistent JSON output
                "num_predict": 1024 # Max tokens to generate
            }
        }
        
        try:
            response = requests.post(self.api_url, headers=headers, json=data)
            response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
            
            # Ollama's /api/generate endpoint returns a JSON object with 'response' key
            response_json = response.json()
            generated_text = response_json.get("response", "").strip()

            # Create a mock response object to fit the existing code's expectation (.text)
            class MockResponse:
                def __init__(self, text):
                    self.text = text
            
            return MockResponse(generated_text)
            
        except requests.exceptions.ConnectionError as e:
            raise ConnectionError(f"Could not connect to Ollama server at {self.api_url}. Is Ollama running? Error: {e}")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error calling Ollama API: {e}")


# Initialize the Ollama Gemma model
model = OllamaGemmaModel(OLLAMA_MODEL_NAME, OLLAMA_API_BASE_URL)


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

def analyze_html_with_gemma(html_content, file_path):
    """
    Sends HTML content to Gemma via Ollama and extracts phishing indicators,
    expecting a structured JSON response by crafting a detailed prompt.
    """
    if not html_content:
        return {"error": "No content to analyze.", "summary_judgment": "N/A"}, "NoContent"

    # Construct the prompt as a single string, including instructions for JSON output
    prompt = f"""
    Analyze the following cleaned HTML content from a webpage to determine if it's a phishing attempt or a legitimate page.
    Provide your analysis as a pure JSON object, without any surrounding markdown formatting (e.g., no ```json at the start or ``` at the end).
    Ensure the JSON is perfectly formatted and valid. If you cannot extract a specific indicator, set its status to false and extracted_data to an empty list.

    JSON Structure:
    {{
        "summary_judgment": "A concise judgment (e.g., 'Phish' or 'Not Phish')",
        "confidence_score": "A score from 0.0 to 1.0 indicating confidence",
        "phishing_indicators": {{
            "brand_impersonation": {{"status": boolean, "extracted_data": [list of strings]}},
            "sensitive_info_request": {{"status": boolean, "extracted_data": [list of strings]}},
            "suspicious_urls_scripts": {{"status": boolean, "extracted_data": [list of strings]}},
            "emotional_language": {{"status": boolean, "extracted_data": [list of strings]}},
            "misspellings_errors": {{"status": boolean, "extracted_data": [list of strings]}},
            "other_indicators": {{"status": boolean, "extracted_data": [list of strings]}}
        }},
        "legitimate_indicators": {{
            "verifiable_details": {{"status": boolean, "extracted_data": [list of strings]}},
            "professional_tone": {{"status": boolean, "extracted_data": [list of strings]}},
            "standard_features": {{"status": boolean, "extracted_data": [list of strings]}},
            "other_indicators": {{"status": boolean, "extracted_data": [list of strings]}}
        }},
        "detailed_explanation": "A comprehensive explanation of the findings and reasoning"
    }}

    Cleaned HTML Content:
    ---
    {html_content}
    ---
    """

    try:
        response = model.generate_content(prompt)
        gemma_response_text = response.text

        # The prompt strongly requests pure JSON. However, models can sometimes
        # add conversational text or markdown. Let's try to find the first '{'
        # and last '}' to isolate the JSON if present.
        json_start = gemma_response_text.find('{')
        json_end = gemma_response_text.rfind('}')

        if json_start != -1 and json_end != -1 and json_end > json_start:
            isolated_json_string = gemma_response_text[json_start : json_end + 1]
        else:
            # If no clear JSON markers, try to strip common markdown patterns
            # as a fallback, though less reliable.
            if gemma_response_text.startswith("```json"):
                isolated_json_string = gemma_response_text[len("```json"):].lstrip('\n')
            elif gemma_response_text.startswith("```"):
                isolated_json_string = gemma_response_text[len("```"):].lstrip('\n')
            else:
                isolated_json_string = gemma_response_text # Assume it's the JSON

            if isolated_json_string.endswith("```"):
                isolated_json_string = isolated_json_string[:-len("```")].rstrip('\n')
            
            # A final trim
            isolated_json_string = isolated_json_string.strip()


        try:
            gemma_response_json = json.loads(isolated_json_string)
            return gemma_response_json, "Success"
        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON from Ollama response for {file_path}. Attempted JSON (cleaned): {isolated_json_string[:500]}...")
            return {"error": "Invalid JSON response from Ollama", "raw_response_cleaned": isolated_json_string[:1000], "summary_judgment": "Parsing Error"}, "Error"
    except ConnectionError as e:
        print(f"Ollama Connection Error for {file_path}: {e}")
        return {"error": f"Ollama Connection Error: {e}", "summary_judgment": "Ollama Not Running"}, "Error"
    except Exception as e:
        print(f"Error calling Ollama API for {file_path}: {e}")
        return {"error": f"Ollama API Error: {e}", "summary_judgment": "API Error"}, "Error"

# --- Main Script ----

if __name__ == "__main__":
    print(f"Starting HTML analysis in {BASE_DIR} using Ollama Gemma {OLLAMA_MODEL_NAME}...")

    phish_files = []
    notphish_files = []

    # --- Step 1: Collect file paths from both categories ---
    print("Collecting file paths...")
    for category_name, file_list in [("Phish", phish_files), ("NotPhish", notphish_files)]:
        category_path = os.path.join(BASE_DIR, category_name)
        if not os.path.isdir(category_path):
            print(f"Warning: Directory '{category_path}' not found. Skipping collection for this category.")
            continue

        try:
            for root, _, files in os.walk(category_path):
                for filename in files:
                    if (filename.endswith(".html") or filename.endswith(".htm")) and len(file_list) < FILE_LIMIT_PER_CATEGORY:
                        file_list.append(os.path.join(root, filename))
            print(f"  Found {len(file_list)} HTML files for '{category_name}'.")
        except Exception as e:
            print(f"Error walking directory {category_path}: {e}")

    # --- Step 2: Interleave the file paths ---
    total_files_to_process = min(len(phish_files), len(notphish_files)) * 2
    if total_files_to_process == 0:
        print("Not enough files in both categories to interleave. Exiting.")
        exit()

    interleaved_files = []
    phish_idx = 0
    notphish_idx = 0

    while phish_idx < len(phish_files) and notphish_idx < len(notphish_files) \
          and len(interleaved_files) < (FILE_LIMIT_PER_CATEGORY * 2):

        # Add a file from Phish
        if phish_idx < len(phish_files) and phish_idx < FILE_LIMIT_PER_CATEGORY:
            interleaved_files.append({"file_path": phish_files[phish_idx], "category": "Phish"})
            phish_idx += 1
        if len(interleaved_files) >= (FILE_LIMIT_PER_CATEGORY * 2): break

        # Add a file from NotPhish
        if notphish_idx < len(notphish_files) and notphish_idx < FILE_LIMIT_PER_CATEGORY:
            interleaved_files.append({"file_path": notphish_files[notphish_idx], "category": "NotPhish"})
            notphish_idx += 1
        if len(interleaved_files) >= (FILE_LIMIT_PER_CATEGORY * 2): break

    print(f"\nStarting interleaved analysis of {len(interleaved_files)} files (up to {FILE_LIMIT_PER_CATEGORY} from each category)...")

    # --- Step 3: Process the interleaved files and save iteratively ---
    csv_exists = os.path.exists(OUTPUT_CSV_FILE)

    for i, file_info in enumerate(interleaved_files):
        file_path = file_info["file_path"]
        category = file_info["category"]
        filename = os.path.basename(file_path)

        print(f"  Analyzing ({i + 1}/{len(interleaved_files)}) [{category}]: {file_path}")

        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                html_content = f.read()

            cleaned_content = clean_html_for_llm(html_content)
            
            # Function name changed for clarity
            gemma_analysis_json, status = analyze_html_with_gemma(cleaned_content, file_path)

            gemma_analysis_str = json.dumps(gemma_analysis_json)

            record_df = pd.DataFrame([{
                "file_path": file_path,
                "original_category": category,
                "gemma_analysis_json": gemma_analysis_str, # Renamed key
                "gemma_status": status, # Renamed key
                "cleaned_html_sample": cleaned_content[:500] + "..." if len(cleaned_content) > 500 else cleaned_content
            }])

            record_df.to_csv(OUTPUT_CSV_FILE, mode='a', header=not csv_exists, index=False)
            csv_exists = True
            print(f"    Saved data for '{filename}' to {OUTPUT_CSV_FILE}")

        except Exception as e:
            print(f"    Critical error processing file {file_path}: {e}. This record will be saved with error info.")
            error_record_df = pd.DataFrame([{
                "file_path": file_path,
                "original_category": category,
                "gemma_analysis_json": json.dumps({"error": f"Critical file processing error: {e}", "summary_judgment": "Critical File Error"}),
                "gemma_status": "Critical File Error",
                "cleaned_html_sample": ""
            }])
            error_record_df.to_csv(OUTPUT_CSV_FILE, mode='a', header=not csv_exists, index=False)
            csv_exists = True
            
    print("\nAnalysis complete. All processed data saved iteratively.")

    if os.path.exists(OUTPUT_CSV_FILE):
        df_final = pd.read_csv(OUTPUT_CSV_FILE)
        print(f"\nFirst 5 rows of the final dataset in '{OUTPUT_CSV_FILE}':\n{df_final.head()}")
    else:
        print(f"\nNo dataset file '{OUTPUT_CSV_FILE}' was created.")