import os
import json
from datetime import datetime

DAILY_TOKEN_LIMIT = 500000 
TOKEN_COUNT_FILE = "groq_token_count.txt"

def update_and_calculate_tokens(response, query, retrieved_nodes):
    """Parses Groq hardware token footprints and tracks daily cumulative caps."""
    p_tokens, c_tokens, q_total = 0, 0, 0
    try:
        if hasattr(response, "additional_kwargs") and "raw" in response.additional_kwargs:
            raw_res = response.additional_kwargs["raw"]
            if hasattr(raw_res, "usage") and raw_res.usage:
                p_tokens = getattr(raw_res.usage, "prompt_tokens", 0)
                c_tokens = getattr(raw_res.usage, "completion_tokens", 0)
                q_total = getattr(raw_res.usage, "total_tokens", 0)
    except Exception:
        pass
        
    if q_total == 0:
        p_tokens = int((len(query) + sum([len(n.text) for n in retrieved_nodes])) / 4)
        c_tokens = int(len(str(response)) / 4)
        q_total = p_tokens + c_tokens

    today_str = datetime.today().strftime('%Y-%m-%d')
    cumulative_tokens = 0
    
    if os.path.exists(TOKEN_COUNT_FILE):
        try:
            with open(TOKEN_COUNT_FILE, 'r') as f:
                data = json.load(f)
                if data.get("date") == today_str:
                    cumulative_tokens = data.get("total_tokens_consumed", 0)
        except Exception:
            pass

    cumulative_tokens += q_total
    try:
        with open(TOKEN_COUNT_FILE, 'w') as f:
            json.dump({"date": today_str, "total_tokens_consumed": cumulative_tokens}, f)
    except Exception:
        pass

    remaining_tokens = max(0, DAILY_TOKEN_LIMIT - cumulative_tokens)
    return {
        "current_total": q_total,
        "current_prompt": p_tokens,
        "current_completion": c_tokens,
        "today_cumulative": cumulative_tokens,
        "today_remaining": remaining_tokens
    }
