"""
WeStar Utility Functions
Common helpers for data loading, LLM calling, and file I/O.
"""
import json
import os
import time
from typing import List, Dict, Any, Optional


# ============= File I/O =============

def load_jsonl(path: str) -> List[Dict]:
    """Load a JSONL file into a list of dicts."""
    results = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def save_jsonl(data: List[Dict], path: str):
    """Save a list of dicts to a JSONL file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def load_json(path: str) -> Any:
    """Load a JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Any, path: str):
    """Save data to a JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


# ============= LLM Calling =============

def call_llm(prompt: str, model: str = "deepseek-r1", 
             temperature: float = 0.7, max_tokens: int = 2048,
             api_base: str = None, api_key: str = None) -> str:
    """
    Call an LLM API for inference.
    
    Supports:
      - OpenAI-compatible APIs (vLLM, DeepSeek, etc.)
      - Modify this function to use your preferred API
    
    Args:
        prompt: Input prompt text
        model: Model name/ID
        temperature: Sampling temperature
        max_tokens: Maximum output tokens
        api_base: API base URL (defaults to env var OPENAI_API_BASE)
        api_key: API key (defaults to env var OPENAI_API_KEY)
    
    Returns:
        Generated text response
    """
    from openai import OpenAI
    
    api_base = api_base or os.environ.get("OPENAI_API_BASE", "http://localhost:8000/v1")
    api_key = api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")
    
    client = OpenAI(base_url=api_base, api_key=api_key)
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            else:
                print(f"[ERROR] LLM call failed after {max_retries} retries: {e}")
                return ""


def call_llm_batch(prompts: List[str], model: str = "deepseek-r1",
                   batch_size: int = 10, **kwargs) -> List[str]:
    """
    Call LLM for a batch of prompts with simple batching.
    
    Args:
        prompts: List of input prompts
        model: Model name
        batch_size: Number of concurrent requests
    
    Returns:
        List of responses
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    results = [""] * len(prompts)
    
    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        future_to_idx = {}
        for idx, prompt in enumerate(prompts):
            future = executor.submit(call_llm, prompt, model, **kwargs)
            future_to_idx[future] = idx
        
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                print(f"[ERROR] Batch item {idx} failed: {e}")
                results[idx] = ""
    
    return results


# ============= Text Processing =============

def truncate_text(text: str, max_length: int = 3000) -> str:
    """Truncate text to max_length characters."""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def clean_llm_response(response: str) -> str:
    """Clean LLM response by removing common artifacts."""
    # Remove thinking tags
    if '</think>' in response:
        response = response.split('</think>')[-1].strip()
    
    # Remove markdown formatting
    response = response.replace('**', '')
    
    return response.strip()
