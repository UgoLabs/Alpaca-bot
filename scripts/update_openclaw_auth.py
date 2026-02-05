import json
import os

path = r'C:\Users\okwum\.openclaw\agents\main\agent\auth-profiles.json'

with open(path, 'r') as f:
    data = json.load(f)

# Add OpenRouter profile
data['profiles']['openrouter:default'] = {
    "type": "token",
    "provider": "openrouter",
    "token": "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"
}

# Update lastGood
data['lastGood']['openrouter'] = "openrouter:default"

with open(path, 'w') as f:
    json.dump(data, f, indent=2)

print("Updated OpenRouter credentials in auth-profiles.json")
