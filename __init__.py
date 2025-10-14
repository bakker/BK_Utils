# Auto-import all Python node files in this folder, with logging
import os
import importlib

folder = os.path.dirname(__file__)

print(f"[BK_Utils] Loading custom nodes from {folder}...")

for filename in os.listdir(folder):
    if filename.endswith(".py") and filename != "__init__.py":
        module_name = filename[:-3]  # strip ".py"
        try:
            module = importlib.import_module(f".{module_name}", package=__name__)
            print(f"[BK_Utils] Imported module: {module_name}")

            # Register node mappings if present
            if hasattr(module, "NODE_CLASS_MAPPINGS"):
                globals().update(module.NODE_CLASS_MAPPINGS)
                print(f"[BK_Utils] Registered NODE_CLASS_MAPPINGS from {module_name}")
            if hasattr(module, "NODE_DISPLAY_NAME_MAPPINGS"):
                globals().update(module.NODE_DISPLAY_NAME_MAPPINGS)
                print(f"[BK_Utils] Registered NODE_DISPLAY_NAME_MAPPINGS from {module_name}")

        except Exception as e:
            print(f"[BK_Utils] Failed to import {module_name}: {e}")

print("[BK_Utils] All modules loaded.\n")