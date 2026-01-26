# My-AI RunPod Launcher

Launch LLMs on RunPod GPUs with a terminal chat interface. Supports multi-GPU tensor parallelism, multiple models, and session recovery.

## Quick Start

**Already have RunPod set up?** Just run:

```bash
export RUNPOD_API_KEY="your_runpod_api_key"
export VLLM_API_KEY="your_chosen_api_key"
python -m my_ai_package.main
```

Select GPU, model, and chat settings - then start chatting.

---

## Prerequisites

### 1. Create a RunPod Account

1. Go to [runpod.io](https://runpod.io) and sign up
2. Add credits to your account (GPU usage is billed per minute)
3. Navigate to **Settings > API Keys**
4. Click **Create API Key** and copy it - this is your `RUNPOD_API_KEY`

### 2. Set Up a Network Volume

The app requires a network volume with vLLM and models pre-installed:

1. In RunPod, go to **Storage > Network Volumes**
2. Create a new volume (50GB+ recommended)
3. Note the volume ID (you'll need to update `launcher.py` with this)
4. Mount the volume to a pod and install:
   - vLLM and dependencies in `/workspace/python_pkgs`
   - Model weights in `/workspace/models`
   - Startup scripts (`start_vllm.sh`, etc.)

### 3. Choose a vLLM API Key

The `VLLM_API_KEY` is a password you choose to secure your vLLM server. Pick any string you like (e.g., `rk_mysecretkey`). This same key is passed to the pod and used by the client.

### 4. Set Environment Variables

**Linux/macOS:**
```bash
export RUNPOD_API_KEY="rpa_xxxxxxxxxxxxxxxxx"
export VLLM_API_KEY="rk_your_chosen_key"
```

**Windows (PowerShell):**
```powershell
$env:RUNPOD_API_KEY="rpa_xxxxxxxxxxxxxxxxx"
$env:VLLM_API_KEY="rk_your_chosen_key"
```

**Windows (Command Prompt):**
```cmd
set RUNPOD_API_KEY=rpa_xxxxxxxxxxxxxxxxx
set VLLM_API_KEY=rk_your_chosen_key
```

---

## Features

### GPU + Model Selection
Choose from available configurations:
- RTX 3090 + Mistral-7B
- RTX PRO 6000 + Mistral-7B
- RTX PRO 6000 + Qwen-32B-AWQ

### Multi-GPU Tensor Parallelism
Select 1-4 GPUs. The app automatically configures `--tensor-parallel-size` for distributed inference across multiple GPUs.

### Chat Configuration
Before chatting, configure:
- **System Prompt**: Helpful, Critic, Engineer, or Creative modes
- **Temperature**: Precise (0.1), Balanced (0.5), or Creative (0.9)
- **Memory Style**: Sliding window (last 10 turns) or Full history

### Session Recovery
If you close the app while a pod is running, relaunch and it will detect the existing pod and offer to reconnect - no need to start a new pod.

### In-Chat Commands
- `/stop` - Terminate the pod and stop billing
- `Ctrl+C` - Exit chat (pod keeps running)

### Auto Pod Detection
The app monitors pod status and exits cleanly if the pod is terminated via RunPod UI.

---

## Usage

```bash
python -m my_ai_package.main
```

**Flow:**
1. Detects existing pod (if any) and offers to reconnect
2. Select GPU type and model
3. Choose number of GPUs (1-4)
4. Confirm to start pod
5. Wait for vLLM API to initialize
6. Configure chat settings (prompt, temperature, memory)
7. Start chatting

---

## Supported Configurations

| GPU | Model | Status |
|-----|-------|--------|
| RTX 3090 | Mistral-7B | Available |
| RTX 4090 | - | Not configured |
| RTX PRO 6000 | Mistral-7B | Available |
| RTX PRO 6000 | Qwen-32B-AWQ | Available |

---

## Project Structure

```
my-ai-runpod-launcher/
├── my_ai_package/
│   ├── main.py         # Entry point
│   ├── ai.py           # GPU selection, chat config, chat UI
│   ├── launcher.py     # Pod creation, status, termination
│   └── config.py       # Runtime config, presets
├── server-files/
│   ├── start_vllm.sh           # Mistral startup script
│   └── start_vllm_qwen32b.sh   # Qwen startup script
├── future_updates.md   # Development roadmap
└── README.md
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `502 from proxy` | Normal during startup. Wait 2-5 minutes for model to load. |
| App hangs on "Waiting for vLLM" | SSH into pod, check `ps aux \| grep vllm` and `nvidia-smi` |
| `RUNPOD_API_KEY not set` | Export the environment variable before running |
| `VLLM_API_KEY not set` | Export the environment variable before running |
| Pod runs but chat fails | API key mismatch - ensure same key is used locally and on pod |
| Tensor parallel fails | Some models don't support TP, or GPU memory insufficient |

### SSH into Pod
```bash
ssh root@<pod-ip> -i ~/.ssh/your_key
```
Or use the RunPod web terminal.

### Verify vLLM is Running
```bash
ps aux | grep vllm
nvidia-smi
curl http://localhost:8000/v1/models -H "Authorization: Bearer $VLLM_API_KEY"
```

---

## Development

See `future_updates.md` for the development roadmap including:
- Reasoning/planning loops
- Judge model for response evaluation
- Full playground mode with custom prompts

---

## Building Executables

### Windows
```
packaging/windows/build.ps1
```

Or if PyInstaller is already installed:
```
pyinstaller --onefile --name my-ai my_ai_package/main.py
```

The executable will be created at `dist/my-ai.exe`.

### Linux
```
packaging/linux/build.sh
```

### macOS
```
packaging/mac/build.sh
```

---

## Running the Windows Executable

### Option 1: Desktop Batch File (Recommended)

Create a file called `my-ai.bat` on your desktop with the following content:

```batch
@echo off
set RUNPOD_API_KEY=rpa_your_runpod_api_key_here
set VLLM_API_KEY=rk_your_chosen_key_here
D:\path\to\my-ai-runpod-launcher\dist\my-ai.exe
pause
```

Replace:
- `rpa_your_runpod_api_key_here` with your actual RunPod API key
- `rk_your_chosen_key_here` with your chosen vLLM API key
- `D:\path\to\...` with the actual path to the executable

Double-click the batch file to launch the app.

### Option 2: Desktop Shortcut

1. Right-click `dist/my-ai.exe` and select **Create shortcut**
2. Move the shortcut to your desktop
3. Right-click the shortcut > **Properties**
4. Set **Start in** to: `D:\path\to\my-ai-runpod-launcher`
5. Click **OK**

Note: With this method, you must set environment variables before running:
- Open Command Prompt
- Run `set RUNPOD_API_KEY=...` and `set VLLM_API_KEY=...`
- Then run the shortcut from the same terminal

### Option 3: System Environment Variables

Set permanent environment variables in Windows:

1. Press `Win + R`, type `sysdm.cpl`, press Enter
2. Go to **Advanced** tab > **Environment Variables**
3. Under **User variables**, click **New**
4. Add `RUNPOD_API_KEY` with your RunPod API key
5. Add `VLLM_API_KEY` with your chosen key
6. Click **OK** and restart any open terminals

Now you can double-click `my-ai.exe` directly.

---

## Running on macOS / Linux

### Build the Executable

```bash
# macOS
./packaging/mac/build.sh

# Linux
./packaging/linux/build.sh
```

Or manually with PyInstaller:
```bash
pip install pyinstaller
pyinstaller --onefile --name my-ai my_ai_package/main.py
```

The executable will be at `dist/my-ai`.

### Option 1: Shell Script Launcher (Recommended)

Create a file called `my-ai-launcher.sh`:

```bash
#!/bin/bash
export RUNPOD_API_KEY="rpa_your_runpod_api_key_here"
export VLLM_API_KEY="rk_your_chosen_key_here"
/path/to/my-ai-runpod-launcher/dist/my-ai
```

Make it executable and run:
```bash
chmod +x my-ai-launcher.sh
./my-ai-launcher.sh
```

### Option 2: Add to Shell Profile

Add to your `~/.bashrc`, `~/.zshrc`, or `~/.bash_profile`:

```bash
export RUNPOD_API_KEY="rpa_your_runpod_api_key_here"
export VLLM_API_KEY="rk_your_chosen_key_here"
alias my-ai="/path/to/my-ai-runpod-launcher/dist/my-ai"
```

Then reload and run:
```bash
source ~/.zshrc  # or ~/.bashrc
my-ai
```

### Option 3: macOS Automator App

1. Open **Automator** and create a new **Application**
2. Add **Run Shell Script** action
3. Paste:
   ```bash
   export RUNPOD_API_KEY="your_key"
   export VLLM_API_KEY="your_key"
   /path/to/dist/my-ai
   ```
4. Set shell to `/bin/bash` and pass input to `stdin`
5. Save as `My-AI.app` to your Applications folder or Desktop
6. Double-click to run (opens in Terminal)
