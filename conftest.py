import sys
from pathlib import Path

# Put the worktree root on sys.path so `import src.calibration_validity` works
sys.path.insert(0, str(Path(__file__).parent))
