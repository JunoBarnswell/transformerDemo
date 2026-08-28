import sys
from pathlib import Path

# Ensure src is in python path
src_dir = Path(__file__).resolve().parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from chatdemo.chat_cli import main

if __name__ == "__main__":
    checkpoint_path = Path(__file__).resolve().parent / "outputs" / "run_rope" / "best.pt"
    
    # If user provided custom arguments, use them; otherwise use greedy decoding defaults
    if len(sys.argv) == 1:
        sys.argv.extend([
            "--checkpoint", str(checkpoint_path),
            "--show-thinking",
            "--temperature", "0.1",
            "--top-k", "1",
            "--max-new-tokens", "100",
            "--recent-window", "4",
            "--compact-threshold", "6",
        ])
    
    main()
