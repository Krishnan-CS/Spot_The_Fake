import os
import google.generativeai as genai
import pandas as pd
from bs4 import BeautifulSoup
import re
import json

API_KEY = "replaced with actual API key while model building"
genai.configure(api_key=API_KEY)

BASE_DIR = "/html_data/training/"

OUTPUT_CSV_FILE = "phishing_dataset.csv"

FILE_LIMIT_PER_CATEGORY = 1000

# Gemini model setup
model = genai.GenerativeModel('gemini-2.5-pro') 


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

def analyze_html_with_gemini(html_content, file_path):
    """
    Sends HTML content to Gemini and extracts phishing indicators,
    expecting a structured JSON response by crafting a detailed prompt.
    """
    if not html_content:
        return {"error": "No content to analyze.", "summary_judgment": "N/A"}, "NoContent"

    # Construct the prompt as a single string, including instructions for JSON output
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

    try:
        response = model.generate_content(prompt)
        gemini_response_text = response.text
        if gemini_response_text.startswith("```json"):
            gemini_response_text = gemini_response_text[len("```json"):].lstrip('\n')
        if gemini_response_text.endswith("```"):
            gemini_response_text = gemini_response_text[:-len("```")].rstrip('\n')

        try:
            gemini_response_json = json.loads(gemini_response_text)
            return gemini_response_json, "Success"
        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON from API response for {file_path}. Raw response (cleaned): {gemini_response_text[:500]}...") # Truncate for log
            return {"error": "Invalid JSON response from API", "raw_response_cleaned": gemini_response_text[:1000], "summary_judgment": "Parsing Error"}, "Error"
    except genai.types.BlockedPromptException as e:
        print(f"Gemini API BlockedPromptException for {file_path}: {e}")
        return {"error": f"Prompt blocked: {e}", "summary_judgment": "Blocked"}, "Blocked"
    except Exception as e:
        print(f"Error calling Gemini API for {file_path}: {e}")
        return {"error": f"Gemini API Error: {e}", "summary_judgment": "API Error"}, "Error"

# --- Main Script ----

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
            
            gemini_analysis_json, status = analyze_html_with_gemini(cleaned_content, file_path)

            gemini_analysis_str = json.dumps(gemini_analysis_json)

            record_df = pd.DataFrame([{
                "file_path": file_path,
                "original_category": category,
                "gemini_analysis_json": gemini_analysis_str,
                "gemini_status": status,
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
                "gemini_analysis_json": json.dumps({"error": f"Critical file processing error: {e}", "summary_judgment": "Critical File Error"}),
                "gemini_status": "Critical File Error",
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
