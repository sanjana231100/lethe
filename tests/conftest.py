import os
import sys
from pathlib import Path

os.environ["MOCK"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
