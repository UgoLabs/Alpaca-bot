
import os

files = [
    r"C:\Users\okwum\AppData\Roaming\npm\node_modules\openclaw\dist\model-selection-BuydNrNr.js",
    r"C:\Users\okwum\AppData\Roaming\npm\node_modules\openclaw\dist\model-selection-qIT4GiGk.js"
]

# We include both the namespaced and non-namespaced versions to ensure matching
patch = """
	{
		id: "google/gemini-1.5-flash-001",
		name: "Gemini 1.5 Flash",
		reasoning: false,
		input: ["text", "image"],
		contextWindow: 1000000,
		maxTokens: 8192,
		privacy: "private"
	},
    {
		id: "gemini-1.5-flash-001",
		name: "Gemini 1.5 Flash (Short)",
		reasoning: false,
		input: ["text", "image"],
		contextWindow: 1000000,
		maxTokens: 8192,
		privacy: "private"
	},
"""

for file_path in files:
    if not os.path.exists(file_path):
        print(f"Skipping {file_path}")
        continue
        
    print(f"Patching {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        if "gemini-1.5-flash-001" in content:
            print("Already patched.")
            continue

        anchor = 'id: "gemini-3-pro-preview",'
        if anchor not in content:
            print(f"Anchor not found in {file_path}")
            continue

        # Find the location of the anchor
        idx = content.find(anchor)
        
        # We want to insert *before* this object to be safe, or after.
        # Let's insert before to avoid messing up comma traversal of the previous item which we don't know.
        # But wait, the list is comma separated. If I insert before, I need to ensure the previous item ends with comma, which it does.
        # But if I insert at the start of THIS object, I need to add a comma to my patch.
        # Let's insert AFTER the anchor object.
        
        # Find the closing brace of the object containing the anchor.
        # Scanning for `},` allows us to handle nested internal braces if any (unlikely here).
        # Simple heuristic: The objects are simple dicts.
        
        end_obj_idx = content.find('},', idx)
        if end_obj_idx == -1:
            print("Could not find end of object")
            continue

        # Insert after `},` (length 2)
        insert_pos = end_obj_idx + 2
        
        new_content = content[:insert_pos] + patch + content[insert_pos:]
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print("Success.")
        
    except Exception as e:
        print(f"Error: {e}")
