import os
import json
import logging
from datetime import datetime

import pandas as pd
import openai
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Constants
INPUT_FILE = "test_input/input.csv"
PROMPT_FILE = "test_input/prompt.txt"
OUTPUT_DIR = "test_results"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_prompt(filepath):
    """Load the system prompt from a file."""
    try:
        with open(filepath, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        logging.error(f"Prompt file not found: {filepath}")
        raise


def normalize_move_date(move_date):
    """Ensure only one move date is returned or return 'N/A'."""
    if isinstance(move_date, list):
        unique_dates = set(move_date)
        return move_date[0] if len(unique_dates) == 1 else "N/A"
    elif isinstance(move_date, str) and "," in move_date:
        dates = set(date.strip() for date in move_date.split(","))
        return dates.pop() if len(dates) == 1 else "N/A"
    return move_date or "N/A"


def extract_attributes(chat_history, prompt):
    """Query OpenAI and extract structured attributes from the response."""
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            temperature=0.1,  # Added temperature parameter
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": chat_history},
            ],
        )
        content = response.choices[0].message.content.strip()
        extracted = json.loads(content)

        email = extracted.get("email", "N/A")
        phone = extracted.get("phone", [])
        if isinstance(phone, str):
            phone = [phone]
        formatted_phone = ", ".join(phone) if phone else "N/A"
        move_date = normalize_move_date(extracted.get("move_date"))

        return f"Email: {email} | Phone: {formatted_phone} | Move date: {move_date}"

    except json.JSONDecodeError:
        logging.warning("Failed to parse JSON response")
        return "Invalid format"
    except Exception as e:
        logging.error(f"OpenAI API error: {e}")
        return "Error during API call"


def evaluate(expected, actual):
    """Case-insensitive comparison of expected vs actual attributes."""
    return str(expected).strip().lower() == str(actual).strip().lower()


def main():
    prompt = load_prompt(PROMPT_FILE)
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        logging.error(f"Input file not found: {INPUT_FILE}")
        return

    if not {"chat_history", "expected_attributes"}.issubset(df.columns):
        raise KeyError("CSV must contain 'chat_history' and 'expected_attributes' columns")

    results = []
    for _, row in df.iterrows():
        chat = row["chat_history"]
        expected = row["expected_attributes"]
        actual = extract_attributes(chat, prompt)
        match = "TRUE" if evaluate(expected, actual) else "FALSE"
        results.append([chat, expected, actual, match])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(OUTPUT_DIR, f"output_{timestamp}.csv")

    results_df = pd.DataFrame(results, columns=["chat_history", "expected_attributes", "actual_attributes", "status"])
    results_df.to_csv(output_path, index=False)
    logging.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
