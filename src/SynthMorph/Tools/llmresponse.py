import re
import json
from typing import Annotated, Optional, Dict, Any, List, TypedDict


def get_structure_response(llm, message, max_retries = 5) -> dict:
    """Directly parse the LLM response and return a dictionary"""
    retries = 0 # Initialize retry count
    while retries < max_retries:
        try:
            response = llm.invoke([message])
            print(f"[INFO] VLM raw response: {response.content}")
            try:
                match = re.search(r'```json\s*([\s\S]*?)\s*```', response.content, re.DOTALL)
                if match:
                    json_content = match.group(1)
                else:
                    json_content = response.content.strip()
                structure_response = json.loads(json_content)
                break
            except json.JSONDecodeError as e:
                print(f"[ERROR] JSON parsing failed: {e}")
                raise e  # Raise exception to trigger retry logic
        except Exception as e:
            retries += 1
            print(f"[ERROR] Parse failed at attempt {retries}: {e}")
            if retries >= max_retries:
                print("[ERROR] Maximum retries reached, returning default value.")
                structure_response = {}
                break
    return structure_response
