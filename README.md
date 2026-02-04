# AGGA-STL
**Repository for LoRA scripts, model training, and optimization tools.**

This repository contains the **Herrscher AGGA Suite**, a set of essential optimization and repair tools for Stable Diffusion (A1111/Forge), specifically designed for limited memory environments like Google Colab.

## 🛠️ Included Tools

### 1. 🛡️ Herrscher Shield (NaN Fixer)
* **Problem:** Black screens when generating images with corrupt SDXL models (Velvette, Pony merges) caused by NaNs in Layer 11.
* **Solution:** This script surgically scans the Text Encoder before every generation and repairs `NaN` or `Inf` values in RAM, preventing the crash without altering the original model file.

### 2. 🚀 AGGA Optimizer (Anti-Crash)
* **Problem:** Google Colab crashes or restarts due to insufficient System RAM (System RAM Crash) when loading heavy models.
* **Solution:** Forces an aggressive memory cleanup (Python GC + Linux malloc_trim) and automatically optimizes A1111 cache settings to keep RAM usage to a minimum.

---

## 📥 Automatic Installation (Google Colab)

To install both tools, follow these steps:

1.  Open your Colab notebook (Automatic1111 or Forge).
2.  Create a **new code cell** after the WebUI installation, but **BEFORE** launching it.
3.  Copy and paste the following code into the cell and run it:

```python
# @title 🧬 Install Herrscher AGGA Suite (Shield + Optimizer)
import os
import requests
from pathlib import Path

# Configuración
GITHUB_USER = "HerrscherAGGA"
REPO_NAME = "AGGA-STL"
BRANCH = "main"

TOOLS = [
    {"remote": "Scripts/herrscher_shield.py", "local": "herrscher_shield.py"},
    {"remote": "Scripts/agga_optimizer.py", "local": "agga_optimizer.py"}
]

possible_paths = [
    Path('/content/stable-diffusion-webui'),                     # Standard A1111
    Path('/content/webui_forge_cu121_torch231/stable-diffusion-webui'), # Forge standard
    Path('/content/A1111'),                                      # Some notebooks
    Path('/content/gdrive/MyDrive/sd/stable-diffusion-webui'),   # Drive installations
    Path('/content/reforge/stable-diffusion-webui')              # Reforge specific
]

WEBUI_PATH = next((p for p in possible_paths if p.exists()), None)

def install_suite():
    if not WEBUI_PATH:
        print("❌ Error: WebUI not found. Run the install cell first.")
        return

    TARGET_DIR = WEBUI_PATH / "scripts"
    os.makedirs(TARGET_DIR, exist_ok=True)
    
    print(f"🚀 Installing Herrscher Suite into: {TARGET_DIR}")

    for tool in TOOLS:
        url = f"https://raw.githubusercontent.com/{GITHUB_USER}/{REPO_NAME}/{BRANCH}/{tool['remote']}"
        
        dest = TARGET_DIR / tool['local']
        
        try:
            r = requests.get(url)
            if r.status_code == 200:
                dest.write_bytes(r.content)
                print(f"   ✅ Installed: {tool['local']}")
            else:
                print(f"   ❌ Failed to download: {tool['local']} (Status: {r.status_code})")
                print(f"      URL intentada: {url}") 
        except Exception as e:
            print(f"   ❌ Error: {e}")

    print("\n✨ SUITE INSTALLED. Restart WebUI if running.")

if __name__ == "__main__":
    install_suite()
```
