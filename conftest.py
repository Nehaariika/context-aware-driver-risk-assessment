"""
conftest.py
Pytest configuration file that fixes import path issues.
Place this in the project root directory (same level as src/ and tests/).
"""

import sys
import os

# Add the project root to Python path so pytest can find 'src' module
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print(f"[conftest.py] Added to sys.path: {project_root}")
