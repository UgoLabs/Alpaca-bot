
import json
import os

config_path = r"C:\Users\okwum\.openclaw\openclaw.json"

try:
    with open(config_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Ensure structure exists
    if "agents" not in data: data["agents"] = {}
    if "defaults" not in data["agents"]: data["agents"]["defaults"] = {}
    if "models" not in data["agents"]["defaults"]: data["agents"]["defaults"]["models"] = {}
    
    # Define the new model
    model_id = "google/gemini-1.5-flash-001"
    model_def = {
        "provider": "google",
        "model": "gemini-1.5-flash-001",
        "context": 1000000,
        "input": ["text", "image"]
    }
    
    data["agents"]["defaults"]["models"][model_id] = model_def
    
    # Write back
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    
    print(f"Successfully added {model_id} to config.")

except Exception as e:
    print(f"Error: {e}")
