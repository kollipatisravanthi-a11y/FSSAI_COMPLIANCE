from __future__ import annotations

import sys
from pathlib import Path

# Allow running as: `python .\scripts\build_vectorstore.py`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fssai_copilot.vectorstore import VectorStoreConfig, build_index_from_folder


def main() -> None:
    regs_folder = Path("data/regulations")
    cfg = VectorStoreConfig()

    count = build_index_from_folder(regs_folder, cfg)
    print(f"Indexed {count} chunks into {cfg.persist_dir}/{cfg.collection}")


if __name__ == "__main__":
    main()
