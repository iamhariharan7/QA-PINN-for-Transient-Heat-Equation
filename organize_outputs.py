import os
import shutil

outputs_dir = "outputs"
history_dir = os.path.join(outputs_dir, "history")
os.makedirs(history_dir, exist_ok=True)

for item in os.listdir(outputs_dir):
    item_path = os.path.join(outputs_dir, item)
    if os.path.isdir(item_path):
        if item not in ["history", "current_output"]:
            dest = os.path.join(history_dir, item)
            # If destination already exists (e.g., moved previously), append something or skip
            if not os.path.exists(dest):
                shutil.move(item_path, dest)
                print(f"Moved {item} to history/")
            else:
                print(f"Skipped {item}, already exists in history/")
                
print("Cleanup complete.")
