<p align="center">
  <img src="icons/icon-256.png" alt="SolvX QuickPod" width="128" height="128">
</p>

<h1 align="center">SolvX QuickPod</h1>

<p align="center">
  <strong>One-click AI chat on RunPod cloud GPUs</strong>
</p>

<p align="center">
  <a href="https://github.com/tradewithmeai/solvx-quickpod/actions/workflows/build.yml">
    <img src="https://github.com/tradewithmeai/solvx-quickpod/actions/workflows/build.yml/badge.svg" alt="Build Status">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT">
  </a>
  <img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+">
</p>

<p align="center">
  <!-- TODO: Add screenshot here -->
  <!-- <img src="assets/screenshot-chat.png" alt="Chat Demo" width="600"> -->
</p>

---

## Features

- **One-Click Launch** - Download, run, chat. No complex setup.
- **GPU Selection** - Choose from 30+ GPUs with live availability, starting at $0.12/hr
- **Cheapest Available** - Auto-select the lowest cost GPU for your VRAM needs
- **Cloud GPU Power** - Run Mistral-7B-Instruct on cloud GPUs from RTX 3070 to H200
- **Guided Onboarding** - First-run wizard walks you through RunPod signup
- **Session Recovery** - Reconnect to running pods automatically
- **Debug Mode** - View raw JSON API exchanges with `/json`
- **Exit Protection** - Prompts to terminate pod on exit to prevent surprise charges
- **Open Source** - All code available, learn how it works

## Quick Start

### Download & Run

1. **Download** the latest release for your platform:
   - [Windows (.exe)](https://github.com/tradewithmeai/solvx-quickpod/releases/latest)
   - [Linux](https://github.com/tradewithmeai/solvx-quickpod/releases/latest)
   - [macOS](https://github.com/tradewithmeai/solvx-quickpod/releases/latest)

2. **Run** the executable - the onboarding wizard will guide you through:
   - Creating a RunPod account (get **$5 free credit** with $10 deposit)
   - Setting up your API key
   - Creating a server password

3. **Chat** - That's it! The app launches a GPU pod and starts your chat session.

### From Source

```bash
git clone https://github.com/tradewithmeai/solvx-quickpod.git
cd solvx-quickpod
pip install -r requirements.txt
python -m solvx_quickpod.main
```

## Cost

Select your GPU at launch based on your budget and VRAM needs:

| GPU | VRAM | Hourly Rate |
|-----|------|-------------|
| RTX A2000 | 6 GB | ~$0.12/hr |
| RTX 3080 | 10 GB | ~$0.19/hr |
| RTX A4000 | 16 GB | ~$0.24/hr |
| RTX 3090 | 24 GB | ~$0.44/hr |
| RTX 4090 | 24 GB | ~$0.69/hr |
| A100 SXM | 80 GB | ~$1.94/hr |

The app queries RunPod for live GPU availability and lets you pick or auto-select the cheapest option. Use `/stop` to terminate the pod and stop billing when you're done.

## Commands

| Command | Description |
|---------|-------------|
| `/help` | Show available commands |
| `/json` | Toggle JSON debug mode (see raw API requests/responses) |
| `/stop` | Terminate pod and stop billing |
| `Ctrl+C` | Exit chat (prompts to terminate pod) |

## Session Recovery

If you close the app while a pod is running:
1. Run the app again
2. It detects your existing pod
3. Choose to reconnect or start fresh

## Building from Source

<details>
<summary>Build Instructions</summary>

### Windows
```powershell
.\packaging\windows\build.ps1
```

### Linux
```bash
./packaging/linux/build.sh
```

### macOS
```bash
./packaging/mac/build.sh
```

The executable will be at `dist/solvx-quickpod`.

</details>

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "502 from proxy" | Normal during startup. Chat will be ready in 8-9 minutes. |
| Pod won't start | Check your RunPod credit balance |
| Connection lost | Pod may have terminated. Run app again to start fresh. |

## How It Works

1. **GPU Selection** - Choose your GPU from live RunPod availability
2. **Pod Creation** - Launches a RunPod GPU instance with vLLM
3. **Model Loading** - Downloads Mistral-7B-Instruct-AWQ from HuggingFace
4. **Chat Interface** - OpenAI-compatible API with streaming responses
5. **Session Logging** - Conversations saved to `~/.myai/chat_logs/`

## License

[MIT](LICENSE) - Use it, modify it, learn from it.

---

<p align="center">
  <strong>New to RunPod?</strong> <a href="https://runpod.io?ref=q04x36mf">Sign up and get $5 free credit</a>
</p>
