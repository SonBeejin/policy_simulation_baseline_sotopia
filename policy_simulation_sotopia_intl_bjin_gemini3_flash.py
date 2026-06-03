from __future__ import annotations

import asyncio
import sys
from pathlib import Path

_BASE_DIR = Path(__file__).resolve().parents[1]
if str(_BASE_DIR) not in sys.path:
    sys.path.insert(0, str(_BASE_DIR))

import policy_simulation_sotopia_intl_bjin_gemini25_flash as sim
from luxia_gemini3_flash_utils import install_sotopia_gemini3

install_sotopia_gemini3(sim)

if __name__ == "__main__":
    asyncio.run(sim.main())
