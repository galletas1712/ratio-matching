from typing import Dict


# Utility: extract the concatenated prompt string up to the last Assistant marker
def extract_anthropic_prompt(prompt_and_response: str) -> str:
    """Extract the anthropic prompt from a prompt and response pair."""
    search_term = "\n\nAssistant:"
    idx = prompt_and_response.rfind(search_term)
    assert idx != -1, f"Prompt and response does not contain '{search_term}'"
    return prompt_and_response[: idx + len(search_term)]


# Utility: split into prompt string, chosen, and rejected continuations
def split_prompt_and_responses(ex: Dict[str, str]) -> tuple[str, str, str]:
    prompt_str = extract_anthropic_prompt(ex["chosen"])
    chosen_resp = ex["chosen"][len(prompt_str) :]
    rejected_resp = ex["rejected"][len(prompt_str) :]
    return prompt_str, chosen_resp, rejected_resp
