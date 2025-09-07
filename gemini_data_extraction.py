import os
import google.generativeai as genai
import pandas as pd
from bs4 import BeautifulSoup
import re
import json
import time # Import time for potential delays

# --- API Key Management ---
# Replace with your actual API keys
API_KEYS = [
    #"AIzaSyAVLhe4g0UZVEQWmZBqdKhjUZ6Z9lm1HEw",
    "AIzaSyCqntYhz-AkkUoAldRj6J35FjVHIQVliy0",
    #"AIzaSyB6hTrWu3auBWiAEMhX83UhOeoXTj1Skz4",
    #"AIzaSyCmenYhWqSyRjJTSOQcizJQ0MvUo9jp2RU",
    #"AIzaSyAPag7llh0k-9hGtkS0ohAETsNm-owoH1A",
    #"AIzaSyCMRLGIygvWrfYISrd8UxlwrUVUAkXPIu8",
    #"AIzaSyC_FH3fklaYm9VAVREIO5-giPO4bZZENnI"
    # Add more API keys as needed
]
current_api_key_index = 0

class APIQuotaExceededError(Exception):
    """Custom exception for when an API key's quota is exceeded or rate-limited."""
    pass

def configure_gemini_api(api_key):
    """Configures the Gemini API with the given key."""
    genai.configure(api_key=api_key)
    print(f"--- Configured Gemini API with key: {api_key}... ---")

# Initialize with the first valid API key
API_KEYS = [key for key in API_KEYS if key and key.strip()]
if API_KEYS:
    configure_gemini_api(API_KEYS[current_api_key_index])
else:
    raise ValueError("No valid API keys provided in the API_KEYS list.")

# =====================
# --- Config/Paths  ---
# =====================
BASE_DIR = "/Users/chandralekhapamidimukkala/Desktop/ISB_Assignments/hackathon/html_data/training/"
OUTPUT_CSV_FILE = "phishing_dataset.csv"
FILE_LIMIT_PER_CATEGORY = 1000

# Use Gemini 2.5 Pro (adjust if needed)
model = genai.GenerativeModel('gemini-2.5-pro')

# Substring that should trigger immediate API-key rotation if found anywhere in error text
QUOTA_ERROR_SUBSTR = "Gemini API Error: 429 You exceeded your current quota"

# =============================
# --- HTML Cleaning Helper  ---
# =============================
def clean_html_for_llm(html_content):
    """
    Cleans HTML content by extracting visible text and removing excessive whitespace,
    making it more digestible for the LLM.
    """
    try:
        soup = BeautifulSoup(html_content, 'html.parser')

        for script_or_style in soup(["script", "style"]):
            script_or_style.extract()

        text = soup.get_text()

        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for phrase in lines if phrase.strip())
        cleaned_text = ' '.join(chunks)

        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

        MAX_TEXT_LENGTH = 15000
        if len(cleaned_text) > MAX_TEXT_LENGTH:
            return cleaned_text[:MAX_TEXT_LENGTH] + "..."
        return cleaned_text
    except Exception as e:
        print(f"Error cleaning HTML: {e}")
        return ""

# ======================================
# --- Core LLM Call + Error Handling  ---
# ======================================
def analyze_html_with_gemini(html_content, file_path):
    """
    Sends HTML content to Gemini and extracts phishing indicators,
    expecting a structured JSON response by crafting a detailed prompt.

    This function raises APIQuotaExceededError to trigger key rotation if:
    - The JSON response includes an "error" containing the quota substring
    - The undecodable raw response contains the quota substring
    - A BlockedPromptException indicates quota/rate-limit/billing
    - Any generic Exception message (unified) contains the quota substring
    """
    if not html_content:
        return {"error": "No content to analyze.", "summary_judgment": "N/A"}, "NoContent"

    prompt = f"""
    Analyze the following cleaned HTML content from a webpage to determine if it's a phishing attempt or a legitimate page.
    Provide your analysis as a pure JSON object, without any surrounding markdown formatting (e.g., no ```json at the start or ``` at the end).

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

    EXPECTED_QUOTA_ERROR_MESSAGE = QUOTA_ERROR_SUBSTR

    try:
        response = model.generate_content(prompt)
        gemini_response_text = response.text

        # Strip accidental code fences
        if gemini_response_text.startswith("```json"):
            gemini_response_text = gemini_response_text[len("```json"):].lstrip('\n')
        if gemini_response_text.endswith("```"):
            gemini_response_text = gemini_response_text[:-len("```")].rstrip('\n')

        try:
            gemini_response_json = json.loads(gemini_response_text)

            # JSON payload might include an error field
            if "error" in gemini_response_json:
                err_text = str(gemini_response_json["error"])
                if EXPECTED_QUOTA_ERROR_MESSAGE in err_text:
                    print(f"Gemini API Quota Exceeded (from JSON response) for {file_path}: {err_text}")
                    raise APIQuotaExceededError(f"Quota exceeded based on JSON error message: {err_text}")

            return gemini_response_json, "Success"

        except json.JSONDecodeError:
            # Raw (undecodable) response includes the quota substring
            if EXPECTED_QUOTA_ERROR_MESSAGE in gemini_response_text:
                print(f"Gemini API Quota Exceeded (from undecodable response) for {file_path}: {EXPECTED_QUOTA_ERROR_MESSAGE}")
                raise APIQuotaExceededError(f"Quota exceeded based on raw response: {EXPECTED_QUOTA_ERROR_MESSAGE}")

            print(f"Warning: Could not decode JSON from API response for {file_path}. Raw response (cleaned): {gemini_response_text[:500]}...")
            return {
                "error": "Invalid JSON response from API",
                "raw_response_cleaned": gemini_response_text[:1000],
                "summary_judgment": "Parsing Error"
            }, "Error"

    except genai.types.BlockedPromptException as e:
        error_message = str(e).lower()
        if "rate limit exceeded" in error_message or "quota" in error_message or "billing" in error_message:
            print(f"Gemini API Quota/Rate Limit Exceeded (from BlockedPromptException) for {file_path}: {e}")
            raise APIQuotaExceededError(f"Quota/Rate limit exceeded based on BlockedPromptException: {e}")
        else:
            print(f"Gemini API BlockedPromptException for {file_path}: {e}")
            return {"error": f"Prompt blocked: {e}", "summary_judgment": "Blocked"}, "Blocked"

    except Exception as e:
        # Normalize any other exception and check for quota substring
        unified_err = f"Gemini API Error: {e}"
        if EXPECTED_QUOTA_ERROR_MESSAGE.lower() in unified_err.lower():
            print(f"Gemini API Quota Exceeded (from Exception) for {file_path}: {unified_err}")
            raise APIQuotaExceededError(unified_err)

        print(f"Error calling Gemini API for {file_path}: {e}")
        return {"error": unified_err, "summary_judgment": "API Error"}, "Error"

# ======================
# --- Main Execution ---
# ======================
if __name__ == "__main__":
    print(f"Starting HTML analysis in {BASE_DIR}...")

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
    total_files_to_consider = min(len(phish_files), len(notphish_files)) * 2
    if total_files_to_consider == 0:
        print("Not enough files in both categories to interleave. Exiting.")
        exit(0)

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

    print(f"\nCollected {len(interleaved_files)} files for initial consideration.")

    # --- Step 2.5: Load existing processed files and filter ---
    existing_file_names = set()
    csv_exists = os.path.exists(OUTPUT_CSV_FILE)

    if csv_exists:
        print(f"'{OUTPUT_CSV_FILE}' found. Loading already processed file names...")
        try:
            existing_df = pd.read_csv(OUTPUT_CSV_FILE, usecols=["file_name"], dtype={'file_name': str})
            if "file_name" in existing_df.columns:
                existing_file_names = set(existing_df["file_name"].dropna().astype(str))
                print(f"  Found {len(existing_file_names)} previously processed files by name.")
            else:
                print(f"  Warning: 'file_name' column not found in '{OUTPUT_CSV_FILE}'. All files will be processed.")
        except Exception as e:
            print(f"  Error loading existing CSV file: {e}. All files will be processed.")
            # Keep csv_exists True so we append without header to avoid duplicates

    else:
        print(f"'{OUTPUT_CSV_FILE}' not found. All files will be processed.")

    files_to_process = []
    for file_info in interleaved_files:
        file_name = os.path.basename(file_info["file_path"])
        if file_name not in existing_file_names:
            files_to_process.append(file_info)
        else:
            print(f"  Skipping '{file_name}' as it already exists in '{OUTPUT_CSV_FILE}'.")

    if len(interleaved_files) > len(files_to_process):
        skipped_count = len(interleaved_files) - len(files_to_process)
        print(f"  Summary: Skipped {skipped_count} files that were already present in '{OUTPUT_CSV_FILE}'.")

    if not files_to_process:
        print("No new files to process. Exiting.")
        exit(0)

    print(f"\nStarting analysis of {len(files_to_process)} new files (up to {FILE_LIMIT_PER_CATEGORY} from each category)...")

    # --- Step 3: Process the filtered files and save iteratively ---
    for i, file_info in enumerate(files_to_process):
        file_path = file_info["file_path"]
        category = file_info["category"]
        filename = os.path.basename(file_path)

        print(f"  Analyzing ({i + 1}/{len(files_to_process)}) [{category}]: {file_path}")

        retries = 0
        max_retries = len(API_KEYS)

        cleaned_content = ""
        gemini_analysis_json = {"error": "Unhandled processing error", "summary_judgment": "Unhandled Error"}
        status = "Unhandled Error"

        while retries < max_retries:
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    html_content = f.read()

                cleaned_content = clean_html_for_llm(html_content)

                gemini_analysis_json, status = analyze_html_with_gemini(cleaned_content, file_path)

                # --- Extra Guard: if the returned payload or status contains the quota substring, rotate key
                if isinstance(gemini_analysis_json, dict):
                    err_txt = (gemini_analysis_json.get("error") or "") + " " + str(status or "")
                    if QUOTA_ERROR_SUBSTR.lower() in err_txt.lower():
                        print(f"Detected quota error in returned payload for {file_path}. Rotating key...")
                        raise APIQuotaExceededError(err_txt)

                # Success path
                break

            except APIQuotaExceededError:
                # Current key exhausted, rotate to next
                if current_api_key_index < len(API_KEYS):
                    print(f"  API key {API_KEYS[current_api_key_index]}... exhausted or rate-limited. Switching to next key.")
                current_api_key_index += 1
                print(f"  Rotated to API key index {current_api_key_index}.")
                if current_api_key_index < len(API_KEYS):
                    configure_gemini_api(API_KEYS[current_api_key_index])
                    time.sleep(5)  # brief backoff
                    retries += 1
                    print(f"  Retrying with new API key (upp0 {retries}/{max_retries})...")
                else:
                    print("  No more API keys available. Stopping processing for current file and marking as API Exhausted.")
                    gemini_analysis_json = {"error": "All API keys exhausted", "summary_judgment": "API Exhausted"}
                    status = "API Exhausted"
                    break

            except Exception as e:
                print(f"    Critical error processing file {file_path}: {e}. This record will be saved with error info.")
                gemini_analysis_json = {"error": f"Critical file processing error: {e}", "summary_judgment": "Critical File Error"}
                status = "Critical File Error"
                break

        # Persist one row per processed file
        gemini_analysis_str = json.dumps(gemini_analysis_json)

        # Ensure header only when file doesn't exist at the time of first write
        write_header = not os.path.exists(OUTPUT_CSV_FILE)

        record_df = pd.DataFrame([{
            "file_path": file_path,
            "file_name": filename,
            "original_category": category,
            "gemini_analysis_json": gemini_analysis_str,
            "gemini_status": status,
            "cleaned_html_sample": cleaned_content[:500] + "..." if cleaned_content and len(cleaned_content) > 500 else (cleaned_content or "")
        }])

        record_df.to_csv(OUTPUT_CSV_FILE, mode='a', header=write_header, index=False)
        print(f"    Saved data for '{filename}' to {OUTPUT_CSV_FILE}")

    print("\nAnalysis complete. All processed data saved iteratively.")

    if os.path.exists(OUTPUT_CSV_FILE):
        try:
            df_final = pd.read_csv(OUTPUT_CSV_FILE)
            print(f"\nFirst 5 rows of the final dataset in '{OUTPUT_CSV_FILE}':\n{df_final.head()}")
        except Exception as e:
            print(f"\nDataset '{OUTPUT_CSV_FILE}' exists but couldn't be previewed: {e}")
    else:
        print(f"\nNo dataset file '{OUTPUT_CSV_FILE}' was created.")