# BK_Utils/__init__.py
from .BK_Utils import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS


# Set web directory for JavaScript files
WEB_DIRECTORY = "./js"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

print(f"[🐈‍⬛ BK_Utils] Loading {len(NODE_CLASS_MAPPINGS)} node(s) from BK_Utils.py:")
for node_name in NODE_CLASS_MAPPINGS:
    display_name = NODE_DISPLAY_NAME_MAPPINGS.get(node_name, "<no display name>")
    print(f"  - {node_name} ({display_name})")

print("[🐈‍⬛ BK_Utils] All modules loaded.\n")
