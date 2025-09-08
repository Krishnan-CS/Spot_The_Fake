import ollama
import os
from bs4 import BeautifulSoup
import re
import json
import requests # Import the requests library for API calls

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

    def analyze_html_with_gemma(self, file_path):
        """
        Sends HTML content to Gemma via Ollama and extracts phishing indicators,
        expecting a structured JSON response by crafting a detailed prompt.
        """
        if not file_path:
            return {"error": "No content to analyze.", "summary_judgment": "N/A"}, "NoContent"

        with open(file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        # Construct the prompt as a single string, including instructions for JSON output
        prompt = f"""
        Analyze the following cleaned HTML content from a webpage to determine if it's a phishing attempt or not.
        Provide your analysis as a pure JSON object, without any surrounding markdown formatting (e.g., no ```json at the start or ``` at the end).
        Ensure the JSON is perfectly formatted and valid. If you cannot extract a specific indicator, set its status to false and extracted_data to an empty list.

        JSON Structure:
        {{
            "summary_judgment": "A concise judgment (e.g., 'Phish' or 'Not Phish')",
            "confidence_score": "A score from 0.0 to 1.0 indicating confidence",
            "phishing_indicators": {{
                "brand_impersonation": {{"status": boolean}},
                "sensitive_info_request": {{"status": boolean}},
                "suspicious_urls_scripts": {{"status": boolean}},
                "emotional_language": {{"status": boolean}},
                "misspellings_errors": {{"status": boolean}},
                "other_indicators": {{"status": boolean}}
            }},
            "legitimate_indicators": {{
                "verifiable_details": {{"status": boolean}},
                "professional_tone": {{"status": boolean}},
                "standard_features": {{"status": boolean}},
                "other_indicators": {{"status": boolean}}
            }},
            "Brief_explanation": "A brief explanation of findings."
        }}

        Cleaned HTML Content:
        ---
        {html_content}
        ---
        """

        try:
            response = self.generate_content(prompt)
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

