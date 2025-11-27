# GitHub Pages Deployment Guide

This directory contains the mjlab interactive demo for GitHub Pages.

## Structure

```
docs/
├── index.html           # Main landing page with embedded Viser viewer
├── viser-client/        # Viser web client (auto-generated)
└── recordings/          # .viser recording files
    └── demo.viser       # Main demo recording
```

## Deployment

The demo is automatically deployed to GitHub Pages via GitHub Actions (`.github/workflows/deploy-demo.yml`).

### Automatic Deployment

- **Weekly**: Runs every Sunday at midnight UTC to keep the demo fresh
- **Manual**: Can be triggered from the Actions tab
- **On push**: Triggers when `record_demo.py` or the workflow file is modified

### Manual Local Build

To build and test locally:

```bash
# 1. Record a demo
uv run python -m mjlab.scripts.record_demo \
  --output-dir docs/recordings \
  --output-name demo \
  --num-steps 500 \
  --num-envs 8

# 2. Build Viser client
uv run viser-build-client --out-dir docs/viser-client

# 3. Serve locally
cd docs
python -m http.server 8000

# 4. Open http://localhost:8000 in your browser
```

## Customization

### Recording Settings

Edit `src/mjlab/scripts/record_demo.py` to customize:
- `num_steps`: Length of recording
- `num_envs`: Number of parallel environments
- `frame_skip`: Frame sampling rate
- `sleep_duration`: Playback speed

### Landing Page

Edit `docs/index.html` to customize the appearance and content.

## Viser Documentation

For more information about Viser embedded visualizations, see:
- https://viser.studio/main/embedded_visualizations/
