# Param Forge

Interactive terminal UI to run Modulette image generation and collect receipts.

## Requirements
- Python 3.9+
- A TTY terminal (macOS Terminal.app, iTerm, etc.)
- Modulette (image generation library)

## Quick start
```bash
git clone git@github.com:kevinshowkat/param_forge.git
cd param_forge
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install python-dotenv

# Option A: clone Modulette next to this repo
#   (../Modulette relative to param_forge)
# git clone <your Modulette repo> ../Modulette
pip install -e ../Modulette

# Option B: if Modulette is already available elsewhere
# pip install -e /path/to/Modulette

python scripts/param_forge.py
```

## API keys
Set the key for the provider you plan to use. You can export env vars or create a local `.env` file.

- OpenAI: `OPENAI_API_KEY`
- Gemini: `GEMINI_API_KEY` (or `GOOGLE_API_KEY`)
- Flux: `BFL_API_KEY` (or `FLUX_API_KEY`)
- Imagen: `GOOGLE_API_KEY` or Vertex credentials (for example `GOOGLE_APPLICATION_CREDENTIALS`)

Tip: If you run with the OpenAI provider and no key is found, the script will prompt you to set one and can save it to `.env`.

## Usage
Interactive (default when no args are provided):
```bash
python scripts/param_forge.py
```

Explicit interactive:
```bash
python scripts/param_forge.py --interactive
```

Non-interactive defaults:
```bash
python scripts/param_forge.py --defaults
```

## Notes
- If curses can’t initialize (TERM issues or small terminal), the script falls back to a raw prompt flow.
- Receipts are stored next to generated images in the output directory.
