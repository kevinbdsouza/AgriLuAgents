import os
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions 
import logging
import yaml
from typing import Union 
import time
from collections import deque
from threading import Lock

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Rate Limiter Implementation ---
class RateLimiter:
    def __init__(self, max_requests, time_window):
        self.max_requests = max_requests
        self.time_window = time_window  # in seconds
        self.requests = deque()
        self.lock = Lock()
    
    def acquire(self):
        with self.lock:
            now = time.time()
            
            # Remove requests older than the time window
            while self.requests and self.requests[0] <= now - self.time_window:
                self.requests.popleft()
            
            # If we've reached the limit, wait until we can make another request
            if len(self.requests) >= self.max_requests:
                wait_time = self.requests[0] + self.time_window - now
                if wait_time > 0:
                    logging.info(f"Rate limit reached. Waiting {wait_time:.2f} seconds...")
                    time.sleep(wait_time)
                    # Recursively call acquire after waiting
                    return self.acquire()
            
            # Add the current request
            self.requests.append(now)
            return True

# Create a global rate limiter instance (15 requests per minute)
gemini_rate_limiter = RateLimiter(max_requests=15, time_window=60)

# --- Configuration Loading ---
def load_config(config_path="config.yaml"):
    """Loads configuration from a YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            logging.info(f"Loaded configuration from {config_path}")
            return config
    except FileNotFoundError:
        logging.error(f"Configuration file not found: {config_path}")
        return {}
    except yaml.YAMLError as e:
        logging.error(f"Error parsing configuration file {config_path}: {e}")
        return {}

# Load config at module level
config = load_config() 
gemini_config = config.get("gemini", {}) # Get gemini section or empty dict

# --- Default Settings (from config or hardcoded fallbacks) ---
DEFAULT_GEMINI_MODEL = gemini_config.get("model_name", "gemini-1.5-flash-latest")
DEFAULT_MAX_TOKENS = gemini_config.get("max_tokens", 1000)
DEFAULT_TEMPERATURE = gemini_config.get("temperature", 0.7)
DEFAULT_TOP_P = gemini_config.get("top_p", 1.0)
DEFAULT_TOP_K = gemini_config.get("top_k", None)

# --- API Key Resolution (Environment variable takes precedence) ---
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    logging.warning("GOOGLE_API_KEY environment variable not set.")
    # Fallback to config file if env var is not set (less secure)
    config_api_key = gemini_config.get("api_key") # Reads from config dict
    if config_api_key and config_api_key != "YOUR_API_KEY_HERE": # Check if it's not the placeholder
        API_KEY = config_api_key
        logging.info("Using API key from config.yaml (Environment variable recommended)")
    else:
        # Log error only if the placeholder is still there or key is missing entirely
        if config_api_key == "YOUR_API_KEY_HERE":
             logging.error("API key placeholder found in config.yaml. Please replace it or set GOOGLE_API_KEY.")
        else:
             logging.error("API key not found in environment variables or config.yaml.")
        # Decide how to handle this: raise error, exit, or allow proceeding without API calls

# --- Gemini Client Initialization ---
gemini_model = None
if API_KEY:
    try:
        genai.configure(api_key=API_KEY)
        gemini_model = genai.GenerativeModel(DEFAULT_GEMINI_MODEL)
        logging.info(f"Gemini client configured with model: {DEFAULT_GEMINI_MODEL}")
    except Exception as e:
        logging.error(f"Failed to configure Gemini client: {e}")
else:
    logging.warning("Cannot initialize Gemini client: API Key is missing.")


def generate_response(
        system_instruction: str,
        prompt: str,
        model_name: str = DEFAULT_GEMINI_MODEL, 
        max_tokens: int = DEFAULT_MAX_TOKENS,    
        temperature: float = DEFAULT_TEMPERATURE, 
        top_p: float = DEFAULT_TOP_P,            
        top_k: Union[int, None] = DEFAULT_TOP_K,
) -> Union[str, None]: 
    """
    Generates a response using the configured Gemini model.

    Args:
        system_instruction: The system message to guide the model's behavior.
        prompt: The user prompt or query.
        model_name: The specific Gemini model to use for this call.
        max_tokens: Maximum number of tokens to generate.
        temperature: Controls randomness (0=deterministic, >1 more random).
        top_p: Nucleus sampling parameter.
        top_k: Top-k sampling parameter.

    Returns:
        The generated text response, or None if an error occurred.
    """
    if not gemini_model:
        logging.error("Gemini client is not configured. Cannot generate response.")
        return None

    # Apply rate limiting
    gemini_rate_limiter.acquire()
    logging.debug("Rate limit check passed, proceeding with API call")

    # Use the model instance associated with the effective model name
    current_model_instance = gemini_model
    effective_model_name = model_name # Start with the requested model

    # If a different model is explicitly requested for the call
    if model_name != DEFAULT_GEMINI_MODEL:
        logging.info(f"Attempting to use specified model for this call: {model_name}")
        try:
            # Check if we already have an instance or create one
            # Note: Caching model instances might be beneficial if switching often
            current_model_instance = genai.GenerativeModel(model_name)
            logging.info(f"Using specified model instance: {model_name}")
        except Exception as e:
            logging.error(f"Failed to initialize specified model {model_name}: {e}. Falling back to default model {DEFAULT_GEMINI_MODEL}.")
            current_model_instance = gemini_model # Fallback to the default global instance
            effective_model_name = DEFAULT_GEMINI_MODEL # Update effective name for logging
    else:
        # Using the default model, log its name
        effective_model_name = DEFAULT_GEMINI_MODEL 
        logging.debug(f"Using default model: {effective_model_name}")

    generation_config = genai.types.GenerationConfig(
        max_output_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    if top_k is not None:
        generation_config.top_k = top_k

    # Construct the content for Gemini API
    contents = []
    if system_instruction:
        full_prompt = f"{system_instruction}\n\n{prompt}"
        contents.append({"role": "user", "parts": [full_prompt]})
    else:
        contents.append({"role": "user", "parts": [prompt]})

    logging.debug(f"Sending prompt to Gemini ({effective_model_name}): {contents}")

    try:
        response = current_model_instance.generate_content(
            contents=contents,
            generation_config=generation_config,
        )
        if not response.candidates:
            logging.warning(f"Gemini response ({effective_model_name}) has no candidates. Check prompt feedback: {response.prompt_feedback}")
            return None
        generated_text = response.text
        # Log the first 100 chars
        logging.debug(f"Received response from Gemini ({effective_model_name}): {generated_text[:100]}...") 
        # Add log for the full response text
        logging.debug(f"Full Gemini response text:\n{generated_text}")
        return generated_text

    except google_exceptions.GoogleAPICallError as e:
        logging.error(f"Gemini API call failed ({effective_model_name}): {e}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred during Gemini generation ({effective_model_name}): {e}")
        return None

# --- Example Usage (updated to show config usage) ---
if __name__ == "__main__":
    
    if not API_KEY:
        print("Error: GOOGLE_API_KEY environment variable not set, and not found in config.yaml. Exiting.")
        exit(1)
         
    if not gemini_model:
        print("Error: Gemini client failed to initialize. Check logs. Exiting.")
        exit(1)

    print(f"--- Using Gemini Model: {DEFAULT_GEMINI_MODEL} (from config) ---")

    # Example system message and prompt
    example_system_message = (
        "You are a helpful assistant specialized in agricultural simulations. "
        "Provide concise and relevant information based on the input."
    )

    # Example prompt
    example_prompt = "Describe the optimal land use strategy for a parcel in region X given climate projection Y."
    # Note: The code to read from prompt.txt is removed for simplicity here,
    # assuming prompts are managed elsewhere or passed directly.

    print(f"--- Sending Prompt ---")
    print(f"System: {example_system_message}")
    print(f"User: {example_prompt}")
    print("-----------------------")

    # Call the generation function (using defaults from config)
    response_text = generate_response(
        system_instruction=example_system_message,
        prompt=example_prompt
        # Can still override config defaults here if needed:
        # model_name="gemini-pro", 
        # temperature=0.8 
    )

    print("--- Gemini Response ---")
    if response_text:
        print(response_text)
    else:
        print("Failed to get a response from Gemini. Check logs.")
    print("----------------------") 
