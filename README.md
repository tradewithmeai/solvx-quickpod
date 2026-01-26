# SolvX QuickPod

One-click LLM chat on RunPod. No setup complexity - just click, wait, chat.

## Get Started

### 1. Create a RunPod Account

**New to RunPod?** Sign up and get $5 free credit:
**https://runpod.io?ref=q04x36mf**

After signing up:
1. Go to **Settings > API Keys**
2. Click **Create API Key** and copy it

### 2. Set Your API Keys

Create a `.env` file or set environment variables:

**Windows (PowerShell):**
```powershell
$env:RUNPOD_API_KEY="your_runpod_api_key"
$env:VLLM_API_KEY="any_password_you_choose"
```

**Linux/macOS:**
```bash
export RUNPOD_API_KEY="your_runpod_api_key"
export VLLM_API_KEY="any_password_you_choose"
```

The `VLLM_API_KEY` is a password you choose - use any string you like.

### 3. Run

```bash
python -m solvx_quickpod.main
```

That's it. The app will:
- Launch an RTX 3090 pod with Mistral-7B
- Wait for the model to load
- Start a chat session

### 4. Chat Commands

- `/stop` - Terminate pod and stop billing
- `/help` - Show available commands
- `Ctrl+C` - Exit chat (pod keeps running)

## Session Recovery

If you close the app while a pod is running, just run the app again - it will detect the existing pod and offer to reconnect.

## Building Executables

### Windows
```
packaging\windows\build.ps1
```

### Linux
```
./packaging/linux/build.sh
```

### macOS
```
./packaging/mac/build.sh
```

The executable will be at `dist/solvx-quickpod`.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `502 from proxy` | Normal during startup. Wait 2-5 minutes for model to load. |
| `RUNPOD_API_KEY not set` | Set the environment variable before running |
| `VLLM_API_KEY not set` | Set the environment variable before running |

---

Don't have RunPod? **Get $5 free credit: https://runpod.io?ref=q04x36mf**
