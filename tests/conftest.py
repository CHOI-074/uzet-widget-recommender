import sys
from pathlib import Path

# `pytest` 를 레포 루트에서 실행하면 src 패키지가 잡히도록.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
