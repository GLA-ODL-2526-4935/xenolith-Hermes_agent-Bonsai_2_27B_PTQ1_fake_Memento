"""Combined-project entry point. No arguments means dry-run, never model loading."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "Hermes_agent"))
from local_extensions.e4b_memory.launch_stack import main

if __name__ == "__main__":
    raise SystemExit(main())
