import sys
from pathlib import Path

# Add the 'app' directory to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent / "app"))

# Import and run the main application
import app.app
