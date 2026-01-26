#!/usr/bin/env python3

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())  # Find and load .env file (searches up directory tree)

import os
import sys
import time
import json
from pathlib import Path
import requests
import httpx
from rich.console import Console
import importlib

from my_ai_package.launcher import create_pod, wait_for_running, wait_for_proxy, write_state, clear_state, is_pod_running, terminate_pod
from my_ai_package import storage


def reload_config_preserve_llm():
    """
    Reload config module while preserving the active LLM port/model selection.
    This picks up POD_ID changes without resetting the user's model choice.
    """
    from my_ai_package import config
    # Save current LLM selection
    saved_port = getattr(config, 'LLM_PORT', 8000)
    saved_model = getattr(config, 'MODEL', None)
    # Reload to pick up POD_ID changes
    importlib.reload(config)
    # Restore LLM selection
    if saved_model:
        config.set_active_llm(saved_port, saved_model)

# ==================== ENVIRONMENT CHECKS ====================

RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
VLLM_API_KEY = os.getenv("VLLM_API_KEY")

if not RUNPOD_API_KEY:
    print("[ERROR] RUNPOD_API_KEY is not set")
    sys.exit(1)

if not VLLM_API_KEY:
    print("[ERROR] VLLM_API_KEY is not set")
    sys.exit(1)

# ==================== CONSTANTS ====================

# GPU + Model configurations
# Each entry is a unique GPU + model combination
GPU_CONFIGS = {
    "RTX 3090 + Mistral-7B": {
        "gpu": "NVIDIA GeForce RTX 3090",
        "model": "/workspace/models/mistral-7b-instruct-awq",
        "script": "start_vllm.sh",
        "available": True,
    },
    "RTX 4090 + Mistral-7B": {
        "gpu": "NVIDIA GeForce RTX 4090",
        "model": "/workspace/models/mistral-7b-instruct-awq",
        "script": "start_vllm.sh",
        "available": True,
    },
    "PRO 6000 + Qwen-32B-AWQ": {
        "gpu": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
        "model": "/workspace/models/qwen2.5-32b-awq",
        "script": "start_vllm_qwen32b.sh",
        "available": True,
    },
    "PRO 6000 + Mistral-7B": {
        "gpu": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
        "model": "/workspace/models/mistral-7b-instruct-awq",
        "script": "start_vllm.sh",
        "available": True,
    },
    # Shell-only pods - bare CUDA Python environment, no auto-start
    "RTX 3090 (Shell Only)": {
        "gpu": "NVIDIA GeForce RTX 3090",
        "model": None,
        "script": None,  # No auto-start script
        "available": True,
        "shell_only": True,
    },
    "RTX 4090 (Shell Only)": {
        "gpu": "NVIDIA GeForce RTX 4090",
        "model": None,
        "script": None,
        "available": True,
        "shell_only": True,
    },
    "PRO 6000 (Shell Only)": {
        "gpu": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
        "model": None,
        "script": None,
        "available": True,
        "shell_only": True,
    },
    # Agentic pod - RTX 3090 with 7 AI services auto-started
    "RTX 3090 (Agentic)": {
        "gpu": "NVIDIA GeForce RTX 3090",
        "model": None,
        "script": None,
        "available": True,
        "agentic": True,
    },
    # Dual LLM pod - RTX 3090 with Mistral-7B + Stable-Code-3B
    "RTX 3090 (Dual LLM)": {
        "gpu": "NVIDIA GeForce RTX 3090",
        "model": "/workspace/models/mistral-7b-instruct-awq",
        "script": None,
        "available": True,
        "dual_llm": True,
    },
    # Full Stack pod - RTX 3090 with both LLMs + 7 Agentic services
    "RTX 3090 (Full Stack)": {
        "gpu": "NVIDIA GeForce RTX 3090",
        "model": "/workspace/models/mistral-7b-instruct-awq",
        "script": None,
        "available": True,
        "full_stack": True,
    },
}

console = Console()
MAX_TURNS = 10

# Local model configuration
LOCAL_MODELS_DIR = r"C:\ai-models\text-generation-webui\models"
DEFAULT_LOCAL_PORT = 5000

# Global state for local server
_local_server = None


# ==================== MODE SELECTION ====================

def select_mode():
    """Select between Local Model and RunPod modes."""
    print("\n=== AI Launcher ===")
    print("Select mode:")
    print("  1. Local PC - Load a model from C:\\ai-models (requires local GPU)")
    print("  2. RunPod Cloud - Launch RTX 3090/4090/PRO 6000 + Mistral/Qwen (default)")

    while True:
        choice = input("\nSelect mode (1-2, Enter for RunPod): ").strip()
        if not choice or choice == "2":
            return "runpod"
        elif choice == "1":
            return "local"
        else:
            print("Please enter 1 or 2")


# ==================== LOCAL MODEL FUNCTIONS ====================

def get_available_models():
    """Scan models directory and return list of available models."""
    if not os.path.exists(LOCAL_MODELS_DIR):
        return []

    models = []
    for name in os.listdir(LOCAL_MODELS_DIR):
        model_path = os.path.join(LOCAL_MODELS_DIR, name)
        if os.path.isdir(model_path):
            files = os.listdir(model_path)
            has_model = any(f.endswith('.safetensors') or f == 'config.json' for f in files)
            if has_model:
                models.append(name)
    return sorted(models)


def wait_for_local_server(url, max_wait=30):
    """Wait for local server to be ready."""
    start = time.time()
    while time.time() - start < max_wait:
        try:
            r = requests.get(url.replace("/v1/chat/completions", "/"), timeout=2)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def run_local_chat(url, model_name="local"):
    """Run non-streaming chat for local model server."""
    print(f"\nChat ready. Connected to: {url}")
    print("Commands: 'quit' to exit, 'new' to clear history\n")

    # Initialize storage session
    session_id = storage.new_session()
    storage.touch_user()

    history = []
    max_history = 20

    while True:
        try:
            user_input = console.input("[bold cyan]You > [/bold cyan]").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n")
            break

        if not user_input:
            continue

        if user_input.lower() in ['quit', 'exit', 'q', '/stop']:
            break

        if user_input.lower() == 'new':
            history = []
            print("[Conversation cleared]\n")
            continue

        messages = history + [{"role": "user", "content": user_input}]
        payload = {
            "model": model_name,
            "messages": messages,
            "max_tokens": 512,
            "temperature": 0.7
        }

        start = time.perf_counter()
        try:
            response = requests.post(url, json=payload, timeout=300)
            elapsed = time.perf_counter() - start

            if response.status_code != 200:
                try:
                    error_data = response.json()
                    error_msg = error_data.get("detail", str(error_data))
                except Exception:
                    error_msg = response.text[:500] if response.text else f"HTTP {response.status_code}"
                console.print(f"[red]Error: {error_msg}[/red]\n")
                continue

            data = response.json()
            reply = data["choices"][0]["message"]["content"]

            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": reply})
            storage.log_message(session_id, "user", user_input)
            storage.log_message(session_id, "assistant", reply)
            if len(history) > max_history:
                history = history[-max_history:]

            console.print(f"\n[bold green]AI >[/bold green] {reply}")
            console.print(f"[dim][{elapsed:.2f}s | {len(history)} msgs][/dim]\n")

        except requests.exceptions.ConnectionError:
            console.print("[red]Error: Connection refused[/red]\n")
        except requests.exceptions.Timeout:
            console.print("[red]Error: Timeout[/red]\n")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]\n")


def mode_local():
    """Run local model mode - load and serve a local model."""
    global _local_server

    print("\n=== LOCAL MODEL SETUP ===")

    models = get_available_models()
    if not models:
        print(f"No models found in: {LOCAL_MODELS_DIR}")
        print("Please download models to this directory first.")
        return

    print("\nAvailable models:")
    for i, name in enumerate(models, 1):
        print(f"  [{i}] {name}")

    while True:
        try:
            choice = input(f"\nSelect model (1-{len(models)}): ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(models):
                model_name = models[idx]
                model_path = os.path.join(LOCAL_MODELS_DIR, model_name)
                break
            else:
                print(f"Please enter a number between 1 and {len(models)}")
        except ValueError:
            print("Please enter a valid number")

    is_gptq = "gptq" in model_name.lower() or "awq" in model_name.lower()
    if is_gptq:
        print(f"\n{model_name} is pre-quantized (GPTQ 4-bit)")
    else:
        print(f"\n{model_name} will load in float16")

    port_input = input(f"Port (default {DEFAULT_LOCAL_PORT}): ").strip()
    port = int(port_input) if port_input else DEFAULT_LOCAL_PORT

    print("\n" + "=" * 50)

    from my_ai_package.server import ModelServer
    _local_server = ModelServer()

    def log(msg):
        print(f"  {msg}", flush=True)

    print("Loading model (this may take a few minutes)...", flush=True)
    print("", flush=True)
    try:
        _local_server.load_model(model_path, use_4bit=False, callback=log)
    except Exception as e:
        print(f"\nError loading model: {e}", flush=True)
        _local_server = None
        return

    print("\nStarting server...", flush=True)
    url = _local_server.start(port=port)
    api_url = f"{url}/v1/chat/completions"

    print("Waiting for server...", end=" ", flush=True)
    if wait_for_local_server(api_url, max_wait=30):
        print("OK")
    else:
        print("FAILED")
        _local_server.stop()
        _local_server = None
        return

    print("=" * 50)
    print(f"Server ready at {url}")
    print("=" * 50)

    chat_mode = select_chat_mode()

    if chat_mode == "agent":
        # run_agent expects base_url, use mode="chat" since local server only has /v1/chat/completions
        run_agent(url, None, model_name, debug=False, mode="chat")
    else:
        run_local_chat(api_url, model_name)

    print("\nStopping server...")
    _local_server.stop()
    _local_server = None
    print("Server stopped.")


# ==================== GPU SELECTION ====================

def select_gpu_config():
    """Prompt user to select GPU + model combo. Returns (gpu_type, gpu_count, model, script, shell_only, agentic, dual_llm, full_stack)."""
    # Filter to only available configs
    available_configs = [
        (name, cfg) for name, cfg in GPU_CONFIGS.items()
        if cfg["available"]
    ]

    print("\n=== GPU + Model Configuration ===")
    print("Available options:")
    for idx, (name, cfg) in enumerate(available_configs, start=1):
        default = " (default)" if idx == 1 else ""
        print(f"  {idx}. {name}{default}")

    while True:
        try:
            choice = input(f"\nSelect option (1-{len(available_configs)}, Enter for default): ").strip()
            if not choice:
                config_idx = 0  # Default to first option
            else:
                config_idx = int(choice) - 1
            if 0 <= config_idx < len(available_configs):
                name, cfg = available_configs[config_idx]
                gpu_type = cfg["gpu"]
                model = cfg["model"]
                script = cfg["script"]
                shell_only = cfg.get("shell_only", False)
                agentic = cfg.get("agentic", False)
                dual_llm = cfg.get("dual_llm", False)
                full_stack = cfg.get("full_stack", False)
                break
            else:
                print(f"Please enter a number between 1 and {len(available_configs)}")
        except ValueError:
            print("Please enter a valid number")

    # GPU count selection disabled for now - default to 1
    # while True:
    #     try:
    #         count = input("Number of GPUs (1-4, default 1): ").strip()
    #         if not count:
    #             gpu_count = 1
    #             break
    #         gpu_count = int(count)
    #         if 1 <= gpu_count <= 4:
    #             break
    #         else:
    #             print("Please enter a number between 1 and 4")
    #     except ValueError:
    #         print("Please enter a valid number")
    gpu_count = 1

    return gpu_type, gpu_count, model, script, shell_only, agentic, dual_llm, full_stack


def select_pricing_type():
    """Select between on-demand and spot (interruptible) pricing."""
    # Pricing selection disabled for now - default to on-demand
    # print("\nPricing Type:")
    # print("  1. On-Demand (default) - Guaranteed, runs until you stop it")
    # print("  2. Spot/Interruptible - Cheaper, but can be interrupted anytime")
    #
    # while True:
    #     choice = input("\nSelect pricing (1-2, Enter for default): ").strip()
    #     if not choice or choice == "1":
    #         return False  # on-demand (not spot)
    #     elif choice == "2":
    #         return True   # spot
    #     else:
    #         print("Please enter 1 or 2")
    return False  # on-demand


def confirm_start(gpu_type, gpu_count, model, shell_only=False, agentic=False, dual_llm=False, full_stack=False, spot=False):
    """Ask user to confirm pod start with selected configuration."""
    print("\n=== Configuration Summary ===")
    print(f"GPU Type: {gpu_type}")
    print(f"GPU Count: {gpu_count}")
    print(f"Pricing: {'Spot (interruptible)' if spot else 'On-Demand'}")
    if shell_only:
        print("Mode: Shell Only (bare CUDA Python environment)")
        print("Mount: /workspace (network volume with vllm, models)")
        print("Ports: 7000-7001, 8000-8008")
    elif agentic:
        print("Mode: Agentic (7 AI services auto-started)")
        print("Services: Embeddings, Reranker, CLIP, BLIP, Memory, TTS, STT")
        print("Mount: /workspace (network volume)")
        print("Ports: 7000-7001, 8000-8008")
    elif dual_llm:
        print("Mode: Dual LLM (two vLLM servers)")
        print("Models: Mistral-7B (port 8000), Stable-Code-3B (port 8001)")
        print("Mount: /workspace (network volume)")
        print("Ports: 7000-7001, 8000-8008")
    elif full_stack:
        print("Mode: Full Stack (2 LLMs + 7 AI services)")
        print("LLMs: Mistral-7B (8000), Stable-Code-3B (8001)")
        print("Services: Reranking (8002), CLIP (8003), BLIP (8004), Memory (8005), TTS (8006), STT (8007), Embedding (8008)")
        print("Startup time: ~13 minutes (5 min setup + 8 min LLM loading)")
        print("Mount: /workspace (network volume)")
    else:
        model_name = model.split("/")[-1] if "/" in model else model
        print(f"Model: {model_name}")
    print("\nNote: Starting a pod will incur costs on your RunPod account.")

    while True:
        choice = input("\nStart pod with these settings? (Y/n, Enter for yes): ").strip().lower()
        if not choice or choice == 'y':
            return True
        elif choice == 'n':
            return False
        else:
            print("Please enter 'y' or 'n'")


# ==================== CHAT CONFIGURATION ====================

def select_chat_config():
    """Prompt user to select system prompt and temperature. Returns (system_prompt, temperature)."""
    from my_ai_package import config
    # Chat config selection disabled for now - use defaults
    return config.SYSTEM_PRESETS["helpful"], config.TEMPERATURE_PRESETS["balanced"]


def select_memory_style():
    """Select memory/history style. Returns max_turns (int or None for full)."""
    from my_ai_package import config
    # Memory style selection disabled for now - use default
    return config.MEMORY_STYLES["sliding"]["max_turns"]


def select_chat_mode():
    """Select between regular chat and agent mode."""
    print("\nChat Mode:")
    print("  1. Regular Chat (default)")
    print("  2. Agent Mode (with websearch)")

    while True:
        choice = input("\nSelect mode (1-2, Enter for default): ").strip()
        if not choice or choice == "1":
            return "chat"
        elif choice == "2":
            return "agent"
        else:
            print("Please enter 1 or 2")


# ==================== POD MANAGEMENT ====================

def check_existing_pod():
    """Check if there's a running pod we can reconnect to."""
    from my_ai_package import config
    reload_config_preserve_llm()

    if not config.POD_ID:
        return None

    if is_pod_running(config.POD_ID):
        return config.POD_ID, config.MODEL

    # Pod exists in state but isn't running - clear stale state
    clear_state()
    return None


def launch_pod(gpu_type, gpu_count, model, script, shell_only=False, agentic=False, dual_llm=False, full_stack=False, spot=False):
    """Launch pod using launcher.py functions and return pod details."""
    print("\n=== Starting Pod ===")

    pod_id = create_pod(gpu_type, gpu_count, script, shell_only=shell_only, agentic=agentic, dual_llm=dual_llm, full_stack=full_stack, spot=spot)
    if not pod_id:
        print("[ERROR] Failed to create pod")
        sys.exit(1)

    # Save state with mode identifier
    if agentic:
        write_state(pod_id, "agentic")
    elif dual_llm:
        write_state(pod_id, "dual-llm")
    elif full_stack:
        write_state(pod_id, "full-stack")
    else:
        write_state(pod_id, model or "shell-only")
    wait_for_running(pod_id)

    base_url = f"https://{pod_id}-8000.proxy.runpod.net"

    return pod_id, base_url


def wait_for_vllm_ready(pod_id):
    """Wait for vLLM API to be fully ready by checking /v1/models endpoint."""
    print("Waiting for vLLM API to be ready...")
    check_count = 0

    while True:
        try:
            r = requests.get(
                f"https://{pod_id}-8000.proxy.runpod.net/v1/models",
                headers={"Authorization": f"Bearer {VLLM_API_KEY}"},
                timeout=10
            )
            if r.status_code == 200:
                print("vLLM API ready ✓")
                return
        except requests.RequestException:
            pass

        # Check pod status every 10 iterations (~30 seconds)
        check_count += 1
        if check_count >= 10:
            check_count = 0
            if not is_pod_running(pod_id):
                print("\nPod terminated. Exiting.")
                sys.exit(0)

        time.sleep(3)


# ==================== CHAT FUNCTIONS ====================

def trim_history(messages, max_turns=10):
    """Trim history based on max_turns. None = keep all."""
    if max_turns is None:
        return messages  # Full history

    max_messages = 1 + max_turns * 2
    if len(messages) > max_messages:
        return [messages[0]] + messages[-(max_messages - 1):]
    return messages


def validate_config():
    """Validate required configuration variables."""
    # Reload config to get latest pod_id
    from my_ai_package import config
    reload_config_preserve_llm()

    missing = []
    if not config.LLM_BASE_URL:
        missing.append("LLM_BASE_URL")
    if not config.API_KEY:
        missing.append("VLLM_API_KEY")
    if not config.MODEL:
        missing.append("MODEL")

    if missing:
        console.print(
            f"[bold red]Missing configuration:[/bold red] {', '.join(missing)}"
        )
        sys.exit(1)


def check_connection():
    """Check model availability via /v1/models endpoint."""
    # Reload config to get latest pod_id
    from my_ai_package import config
    reload_config_preserve_llm()

    url = config.LLM_BASE_URL.rstrip("/") + "/v1/models"

    console.print(f"[dim]Checking model availability on port {config.LLM_PORT}...[/dim]")
    try:
        r = httpx.get(
            url,
            headers={"Authorization": f"Bearer {config.API_KEY}"},
            timeout=10,
        )
        r.raise_for_status()
    except Exception as e:
        console.print(
            "\n[bold red]Model not reachable.[/bold red]\n"
            "• Pod may still be starting\n"
            "• API key may be wrong\n"
            f"• Port {config.LLM_PORT} not ready\n\n"
            f"[dim]{e}[/dim]"
        )
        sys.exit(1)


def run_chat(system_prompt=None, temperature=None, max_turns=10):
    """Run the interactive chat interface with streaming responses."""
    # Reload config to get latest pod_id
    from my_ai_package import config
    reload_config_preserve_llm()

    validate_config()
    check_connection()

    # Use provided values or defaults
    if system_prompt is None:
        system_prompt = config.SYSTEM_PROMPT
    if temperature is None:
        temperature = 0.5  # balanced default

    pod_id = config.POD_ID

    # Initialize storage session FIRST (needed for memory context)
    session_id = storage.new_session()
    storage.touch_user()

    # Inject memory context if enabled (BEFORE messages list)
    memory_worker = None
    if config.MEMORY_ENABLED:
        try:
            from my_ai_package.memory_context import build_context_pack
            memory_pack = build_context_pack(session_id)
            if memory_pack:
                system_prompt = f"{memory_pack}\n\n{system_prompt}"
        except Exception:
            pass  # Graceful degradation

        # Start memory worker in background
        try:
            from my_ai_package.memory_services import MemoryServices
            from my_ai_package.memory_worker import MemoryWorker
            services = MemoryServices(pod_id, config.API_KEY)
            memory_worker = MemoryWorker(session_id, services)
            memory_worker.start_background()
        except Exception:
            pass  # Worker failed to start, chat continues

    # Now create messages and log (AFTER memory injection)
    messages = [{"role": "system", "content": system_prompt.strip()}]
    storage.log_message(session_id, "system", system_prompt.strip())

    # Pod check timing
    last_pod_check = time.time()
    POD_CHECK_INTERVAL = 10  # seconds

    # Debug toggle
    show_json = False

    # Model toggle (primary = Mistral on 8000, small = Stable-Code on 8001)
    # Track which model is active (config is the source of truth)
    using_small_model = (config.LLM_PORT == config.SMALL_LLM_PORT)

    # Orchestrator mode toggle
    use_orchestrator = False
    orchestrator = None
    try:
        from my_ai_package.orchestrator import Orchestrator
        from my_ai_package.memory_services import MemoryServices
        from my_ai_package.tools.registry import registry
        from my_ai_package.tools.websearch import websearch
        # Register websearch tool
        if not registry.has_tool("websearch"):
            registry.register("websearch", websearch)
        # Pre-create orchestrator (lazy - only used if toggled on)
        orch_services = MemoryServices(pod_id, config.API_KEY)
        orchestrator = Orchestrator(orch_services, registry, system_prompt)
    except Exception:
        pass  # Orchestrator not available

    # Show active settings
    console.print(f"[dim]System: {system_prompt[:50]}{'...' if len(system_prompt) > 50 else ''}[/dim]")
    console.print(f"[dim]Temperature: {temperature}[/dim]")
    console.print(f"[dim]Memory: {'Full history' if max_turns is None else f'Last {max_turns} turns'}[/dim]\n")
    console.print("[bold]AI chat started. Commands: /stop, /json, /memory, /orchestrator, /judge, /training. Ctrl+C to exit.[/bold]\n")

    while True:
        try:
            # Check pod status periodically
            if time.time() - last_pod_check > POD_CHECK_INTERVAL:
                if not is_pod_running(pod_id):
                    console.print("\n[dim]Pod terminated. Exiting.[/dim]")
                    sys.exit(0)
                last_pod_check = time.time()

            user_input = console.input("[bold cyan]You > [/bold cyan]").strip()
            if not user_input:
                continue

            # Show command help
            if user_input.lower() == "/help":
                console.print("[dim]Available commands:[/dim]")
                console.print("[dim]  /stop         - Terminate pod and exit[/dim]")
                console.print("[dim]  /json         - Toggle JSON debug output[/dim]")
                console.print("[dim]  /memory       - Show extracted memory context[/dim]")
                console.print("[dim]  /memory-test  - Manually trigger memory extraction[/dim]")
                console.print("[dim]  /orchestrator - Toggle orchestrator mode (enables tools)[/dim]")
                console.print("[dim]  /judge <criteria> - Evaluate last response with Stable-Code judge[/dim]")
                console.print("[dim]  /training     - View critic disagreement training data[/dim]")
                console.print("[dim]  /help         - Show this help[/dim]\n")
                continue

            # Handle /stop command
            if user_input.lower() == "/stop":
                confirm = input("Terminate pod? This will stop billing. (y/n): ").strip().lower()
                if confirm == 'y':
                    if terminate_pod(pod_id):
                        console.print("[bold]Pod terminated.[/bold]")
                        sys.exit(0)
                    else:
                        console.print("[red]Failed to terminate pod.[/red]")
                continue

            # Handle /json command
            if user_input.lower() == "/json":
                show_json = not show_json
                console.print(f"[dim]JSON debug: {'ON' if show_json else 'OFF'}[/dim]\n")
                continue

            # Handle /memory command
            if user_input.lower() == "/memory":
                try:
                    from my_ai_package.memory_context import build_context_pack
                    memory_pack = build_context_pack(session_id)
                    if memory_pack:
                        console.print(f"[dim]{memory_pack}[/dim]\n")
                    else:
                        console.print("[dim]No memory stored for this session.[/dim]\n")
                except Exception as e:
                    console.print(f"[dim]Memory unavailable: {e}[/dim]\n")
                continue

            # Handle /memory-test command (manually trigger worker batch)
            if user_input.lower() == "/memory-test":
                try:
                    from my_ai_package.memory_ingest import get_events, get_offset
                    from my_ai_package.memory_worker import process_session
                    from my_ai_package.memory_services import MemoryServices

                    # Show pending events
                    offset = get_offset(session_id)
                    events, _ = get_events(session_id, since_offset=offset)
                    console.print(f"[dim]Session: {session_id}[/dim]")
                    console.print(f"[dim]Pending events: {len(events)}[/dim]")

                    if not events:
                        console.print("[dim]No new events to process.[/dim]\n")
                        continue

                    # Show event summary
                    for i, e in enumerate(events[:5]):
                        etype = e.get('type', '?')
                        content = e.get('payload', {}).get('content', '')[:50]
                        console.print(f"[dim]  [{i}] {etype}: {content}...[/dim]")
                    if len(events) > 5:
                        console.print(f"[dim]  ... and {len(events) - 5} more[/dim]")

                    # Try to process
                    console.print("[dim]Processing batch (calling LLM)...[/dim]")
                    services = MemoryServices(pod_id, config.API_KEY)
                    result = process_session(session_id, services)

                    if result:
                        console.print("[green]Memory extraction successful![/green]")
                        console.print(f"[dim]Summary: {result.get('episode_summary', 'N/A')[:100]}[/dim]")
                        console.print(f"[dim]Facts: {len(result.get('facts', []))}[/dim]")
                        console.print(f"[dim]Tasks: {len(result.get('open_tasks', []))}[/dim]")
                    else:
                        console.print("[yellow]Processing returned None (LLM may be unavailable)[/yellow]")
                except Exception as e:
                    console.print(f"[red]Memory test failed: {e}[/red]")
                console.print()
                continue

            # Handle /model command - DISABLED: reserved for future coding assistant
            # Stable-Code is now judge-only, not for direct chat
            # if user_input.lower() == "/model":
            #     using_small_model = not using_small_model
            #     if using_small_model:
            #         config.set_active_llm(config.SMALL_LLM_PORT, config.SMALL_MODEL)
            #         console.print(f"[dim]Switched to Stable-Code-3B (port {config.LLM_PORT})[/dim]\n")
            #     else:
            #         config.set_active_llm(config.PRIMARY_LLM_PORT, config.PRIMARY_MODEL)
            #         console.print(f"[dim]Switched to Mistral-7B (port {config.LLM_PORT})[/dim]\n")
            #     continue

            # Handle /orchestrator command - toggle orchestrator mode
            if user_input.lower() == "/orchestrator":
                if orchestrator:
                    use_orchestrator = not use_orchestrator
                    if use_orchestrator:
                        console.print("[dim]Orchestrator mode: ON (tools enabled)[/dim]\n")
                    else:
                        console.print("[dim]Orchestrator mode: OFF (direct chat)[/dim]\n")
                else:
                    console.print("[dim]Orchestrator not available[/dim]\n")
                continue

            # Handle /judge command - evaluate last assistant response with Stable-Code
            # Judge invocation is EXPLICIT ONLY - never auto-routed
            if user_input.lower().startswith("/judge"):
                parts = user_input.split(maxsplit=1)
                criteria = parts[1] if len(parts) > 1 else "Is the response helpful and accurate?"

                # Get last assistant message
                last_assistant = None
                for msg in reversed(messages):
                    if msg["role"] == "assistant":
                        last_assistant = msg["content"]
                        break

                if not last_assistant:
                    console.print("[dim]No assistant response to judge.[/dim]\n")
                    continue

                from my_ai_package.judge import judge as run_judge
                verdict = run_judge(criteria, last_assistant, pod_id, config.API_KEY)

                if verdict:
                    status = "[green]PASS[/green]" if verdict.pass_ else "[red]FAIL[/red]"
                    console.print(f"[dim]Verdict: {status}[/dim]")
                    console.print(f"[dim]Score: {verdict.score}[/dim]")
                    console.print(f"[dim]Summary: {verdict.summary}[/dim]")
                    if verdict.issues:
                        console.print(f"[dim]Issues: {', '.join(verdict.issues)}[/dim]")
                else:
                    console.print("[red]Judge evaluation failed (check logs)[/red]")
                console.print()
                continue

            # Handle /training command - view critic training data
            if user_input.lower() == "/training":
                from my_ai_package.orchestrator import summarize_training_data, CRITIC_TRAINING_FILE
                summary = summarize_training_data()
                console.print("[dim]─── CRITIC TRAINING DATA ───[/dim]")
                console.print(f"[dim]File: {CRITIC_TRAINING_FILE}[/dim]")
                console.print(f"[dim]Total disagreements: {summary.get('total', 0)}[/dim]")
                console.print(f"[dim]Fallbacks applied: {summary.get('fallback_applied', 0)}[/dim]")
                if summary.get("by_proposed_action"):
                    console.print(f"[dim]By proposed action: {summary['by_proposed_action']}[/dim]")
                if summary.get("recent_inputs"):
                    console.print("[dim]Recent inputs (for pattern analysis):[/dim]")
                    for inp in summary["recent_inputs"][:5]:
                        console.print(f"[dim]  - {inp}[/dim]")
                console.print()
                continue

            messages.append({"role": "user", "content": user_input})
            storage.log_message(session_id, "user", user_input)
            try:
                from my_ai_package.memory_ingest import emit_event
                emit_event(session_id, "chat.user", {"content": user_input})
            except Exception:
                pass  # Graceful degradation - don't break chat if memory fails
            messages = trim_history(messages, max_turns)

            # Start timing from prompt send
            request_start = time.time()

            # Orchestrator mode - route through orchestrator instead of direct LLM
            if use_orchestrator and orchestrator:
                try:
                    assistant_text = orchestrator.run(session_id, user_input)
                    elapsed = time.time() - request_start
                    console.print(f"\n[bold green]AI >[/bold green] {assistant_text}")
                    console.print(f"[dim]({elapsed:.1f}s)[/dim]\n")
                    messages.append({"role": "assistant", "content": assistant_text})
                    storage.log_message(session_id, "assistant", assistant_text)
                    try:
                        emit_event(session_id, "chat.assistant", {"content": assistant_text})
                    except Exception:
                        pass
                except Exception as e:
                    console.print(f"\n[red]Orchestrator error: {e}[/red]\n")
                continue

            payload = {
                "model": config.MODEL,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": 500,
                "stream": True,
            }

            # Show request JSON if enabled
            if show_json:
                console.print("[dim]─── REQUEST ───[/dim]")
                console.print(f"[dim]POST {config.LLM_BASE_URL.rstrip('/')}/v1/chat/completions[/dim]")
                console.print(f"[dim]{json.dumps(payload, indent=2)}[/dim]\n")

            headers = {
                "Authorization": f"Bearer {config.API_KEY}",
                "Content-Type": "application/json",
            }

            console.print("\n[bold green]AI >[/bold green] ", end="")
            assistant_text = ""

            with httpx.Client(timeout=None) as client:
                with client.stream(
                    "POST",
                    config.LLM_BASE_URL.rstrip("/") + "/v1/chat/completions",
                    headers=headers,
                    json=payload,
                ) as response:

                    if response.status_code != 200:
                        console.print(
                            f"\n[bold red]HTTP {response.status_code}[/bold red]"
                        )
                        continue

                    for line in response.iter_lines():
                        if not line:
                            continue

                        if not line.startswith("data: "):
                            continue

                        data = line[6:]
                        if data == "[DONE]":
                            break

                        try:
                            chunk = json.loads(data)
                            delta = chunk["choices"][0]["delta"]
                            if "content" in delta:
                                token = delta["content"]
                                assistant_text += token
                                console.print(token, end="", soft_wrap=True)
                        except Exception:
                            continue

            elapsed = time.time() - request_start
            console.print(f"\n[dim]({elapsed:.1f}s)[/dim]\n")

            # Show response JSON if enabled
            if show_json and assistant_text:
                response_obj = {
                    "choices": [{"message": {"role": "assistant", "content": assistant_text}}],
                    "model": config.MODEL,
                }
                console.print("[dim]─── RESPONSE ───[/dim]")
                console.print(f"[dim]{json.dumps(response_obj, indent=2)}[/dim]\n")

            if assistant_text:
                messages.append(
                    {"role": "assistant", "content": assistant_text}
                )
                storage.log_message(session_id, "assistant", assistant_text)
                try:
                    from my_ai_package.memory_ingest import emit_event
                    emit_event(session_id, "chat.assistant", {"content": assistant_text})
                except Exception:
                    pass  # Graceful degradation
                messages = trim_history(messages, max_turns)

        except KeyboardInterrupt:
            if memory_worker:
                memory_worker.stop()
            console.print("\n[bold]Session ended.[/bold]")
            sys.exit(0)

        except (httpx.ConnectError, httpx.RemoteProtocolError, httpx.ReadError) as e:
            # Connection error - check if pod is still running
            if not is_pod_running(pod_id):
                console.print("\n[dim]Pod terminated. Exiting.[/dim]")
                sys.exit(0)
            else:
                console.print(f"\n[bold red]Connection error:[/bold red] {e}")
                continue


def run_agent(base_url, api_key, model, debug=False, mode="completion"):
    """Run the agent interface with tool support.

    Args:
        base_url: Server base URL (e.g., "http://localhost:5000")
        api_key: API key for authentication (None for local servers)
        model: Model name/path
        debug: Enable debug output
        mode: "completion" for /v1/completions (default), "chat" for /v1/chat/completions
    """
    from my_ai_package import config
    from my_ai_package.controller import Controller

    controller = Controller(base_url, api_key, model, mode=mode)

    # Debug toggles
    show_json = False
    show_debug = debug

    # Initialize storage session
    session_id = storage.new_session()
    storage.touch_user()

    # Start memory worker in background if enabled
    memory_worker = None
    if hasattr(config, 'MEMORY_ENABLED') and config.MEMORY_ENABLED and base_url:
        try:
            from my_ai_package.memory_services import MemoryServices
            from my_ai_package.memory_worker import MemoryWorker
            # Extract pod_id from base_url
            import re
            match = re.search(r'https://([^-]+)-', base_url)
            if match:
                pod_id = match.group(1)
                services = MemoryServices(pod_id, api_key or config.API_KEY)
                memory_worker = MemoryWorker(session_id, services)
                memory_worker.start_background()
        except Exception:
            pass  # Worker failed to start, agent continues

    console.print("[bold]Agent mode started. Commands: /stop, /json, /debug[/bold]")
    console.print("[dim]Tools available: websearch[/dim]\n")

    while True:
        try:
            user_input = console.input("[bold cyan]You > [/bold cyan]").strip()
            if not user_input:
                continue
            if user_input.lower() == "/stop":
                if memory_worker:
                    memory_worker.stop()
                break

            # Handle /json command
            if user_input.lower() == "/json":
                show_json = not show_json
                console.print(f"[dim]JSON debug: {'ON' if show_json else 'OFF'}[/dim]\n")
                continue

            # Handle /debug command
            if user_input.lower() == "/debug":
                show_debug = not show_debug
                console.print(f"[dim]Tool debug: {'ON' if show_debug else 'OFF'}[/dim]\n")
                continue

            # Log user message
            storage.log_message(session_id, "user", user_input)
            try:
                from my_ai_package.memory_ingest import emit_event
                emit_event(session_id, "chat.user", {"content": user_input})
            except Exception:
                pass  # Graceful degradation

            # Show request if enabled
            if show_json:
                request_obj = {
                    "base_url": base_url,
                    "model": model,
                    "mode": mode,
                    "user_input": user_input,
                }
                console.print("[dim]─── REQUEST ───[/dim]")
                console.print(f"[dim]{json.dumps(request_obj, indent=2)}[/dim]\n")

            result = controller.run(user_input, debug=show_debug, session_id=session_id)

            # Show response if enabled
            if show_json:
                response_obj = {
                    "type": "final",
                    "content": result.content,
                }
                console.print("[dim]─── RESPONSE ───[/dim]")
                console.print(f"[dim]{json.dumps(response_obj, indent=2)}[/dim]\n")

            # Log agent response
            storage.log_message(session_id, "assistant", result.content)
            try:
                from my_ai_package.memory_ingest import emit_event
                emit_event(session_id, "chat.assistant", {"content": result.content})
            except Exception:
                pass  # Graceful degradation

            console.print(f"\n[bold green]Agent >[/bold green] {result.content}\n")

        except KeyboardInterrupt:
            if memory_worker:
                memory_worker.stop()
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


# ==================== MAIN ORCHESTRATION ====================

def main():
    """Main function orchestrating the complete user flow."""
    try:
        # Step 0: Select mode (Local or RunPod)
        mode = select_mode()

        if mode == "local":
            mode_local()
            return

        # RunPod mode continues below...

        # Step 1: Check for existing running pod
        existing = check_existing_pod()
        if existing:
            pod_id, model = existing
            is_shell_only = (model == "shell-only")
            is_agentic = (model == "agentic")
            is_dual_llm = (model == "dual-llm")
            is_full_stack = (model == "full-stack")
            model_name = model.split("/")[-1] if model and "/" in model else (model or "unknown")
            print(f"\n=== Existing Pod Found ===")
            print(f"Pod ID: {pod_id}")
            if is_shell_only:
                print("Mode: Shell Only")
            elif is_agentic:
                print("Mode: Agentic (7 AI services)")
            elif is_dual_llm:
                print("Mode: Dual LLM (Mistral-7B + Stable-Code-3B)")
            elif is_full_stack:
                print("Mode: Full Stack (2 LLMs + 7 AI services)")
            else:
                print(f"Model: {model_name}")

            while True:
                choice = input("\nReconnect to this pod? (y/n): ").strip().lower()
                if choice == 'y':
                    if is_shell_only:
                        # Shell-only: just show connection info
                        print("\n=== Shell-Only Pod ===")
                        print(f"Pod ID: {pod_id}")
                        print(f"SSH: runpodctl connect {pod_id}")
                        print(f"Web Terminal: https://{pod_id}-7860.proxy.runpod.net")
                        print(f"API Port: https://{pod_id}-8000.proxy.runpod.net")
                        print(f"\nMount: /workspace (network volume with vllm, models)")
                        print("\nUse /stop to terminate when done.")
                        input("\nPress Enter to exit...")
                        return
                    elif is_agentic:
                        # Agentic: show all service endpoints
                        print("\n=== Agentic Pod ===")
                        print(f"Pod ID: {pod_id}")
                        print(f"SSH: runpodctl connect {pod_id}")
                        print(f"\n7 AI Services:")
                        print(f"  [1] Embeddings: https://{pod_id}-8000.proxy.runpod.net")
                        print(f"  [2] Reranker:   https://{pod_id}-8001.proxy.runpod.net")
                        print(f"  [3] CLIP:       https://{pod_id}-8002.proxy.runpod.net")
                        print(f"  [4] BLIP:       https://{pod_id}-8003.proxy.runpod.net")
                        print(f"  [5] Memory:     https://{pod_id}-8004.proxy.runpod.net")
                        print(f"  [6] TTS:        https://{pod_id}-8005.proxy.runpod.net")
                        print(f"  [7] STT:        https://{pod_id}-8006.proxy.runpod.net")
                        print(f"\nMount: /workspace (network volume)")
                        print("\nUse /stop to terminate when done.")
                        input("\nPress Enter to exit...")
                        return
                    elif is_dual_llm:
                        # Dual LLM: show endpoints, wait for main model, then chat
                        print("\n=== Dual LLM Pod ===")
                        print(f"Pod ID: {pod_id}")
                        print(f"SSH: runpodctl connect {pod_id}")
                        print(f"\nLLM Endpoints:")
                        print(f"  [1] Mistral-7B:     https://{pod_id}-8000.proxy.runpod.net")
                        print(f"  [2] Stable-Code-3B: https://{pod_id}-8001.proxy.runpod.net")
                        print("\nReconnecting to Mistral-7B...")
                        wait_for_vllm_ready(pod_id)

                        # Chat Configuration
                        system_prompt, temperature = select_chat_config()
                        max_turns = select_memory_style()
                        chat_mode = select_chat_mode()

                        print("\n=== Reconnected ===")
                        print()
                        if chat_mode == "agent":
                            from my_ai_package import config
                            reload_config_preserve_llm()
                            run_agent(config.LLM_BASE_URL, config.API_KEY, config.MODEL, debug=False)
                        else:
                            run_chat(system_prompt, temperature, max_turns)
                        return
                    elif is_full_stack:
                        # Full Stack: show all 9 endpoints, wait for main model, then chat
                        print("\n=== Full Stack Pod ===")
                        print(f"Pod ID: {pod_id}")
                        print(f"SSH: runpodctl connect {pod_id}")
                        print(f"\nAll 9 Services:")
                        print(f"  [1] Mistral-7B:     https://{pod_id}-8000.proxy.runpod.net")
                        print(f"  [2] Stable-Code-3B: https://{pod_id}-8001.proxy.runpod.net")
                        print(f"  [3] Reranking:      https://{pod_id}-8002.proxy.runpod.net")
                        print(f"  [4] CLIP:           https://{pod_id}-8003.proxy.runpod.net")
                        print(f"  [5] BLIP:           https://{pod_id}-8004.proxy.runpod.net")
                        print(f"  [6] Memory:         https://{pod_id}-8005.proxy.runpod.net")
                        print(f"  [7] TTS:            https://{pod_id}-8006.proxy.runpod.net")
                        print(f"  [8] STT:            https://{pod_id}-8007.proxy.runpod.net")
                        print(f"  [9] Embedding:      https://{pod_id}-8008.proxy.runpod.net")
                        print("\nReconnecting to Mistral-7B...")
                        wait_for_vllm_ready(pod_id)

                        # Chat Configuration
                        system_prompt, temperature = select_chat_config()
                        max_turns = select_memory_style()
                        chat_mode = select_chat_mode()

                        print("\n=== Reconnected ===")
                        print()
                        if chat_mode == "agent":
                            from my_ai_package import config
                            reload_config_preserve_llm()
                            run_agent(config.LLM_BASE_URL, config.API_KEY, config.MODEL, debug=False)
                        else:
                            run_chat(system_prompt, temperature, max_turns)
                        return
                    else:
                        # Skip pod creation, go straight to vLLM check
                        print("\nReconnecting...")
                        wait_for_vllm_ready(pod_id)

                        # Chat Configuration
                        system_prompt, temperature = select_chat_config()
                        max_turns = select_memory_style()
                        chat_mode = select_chat_mode()

                        print("\n=== Reconnected ===")
                        print()
                        if chat_mode == "agent":
                            from my_ai_package import config
                            reload_config_preserve_llm()
                            run_agent(config.LLM_BASE_URL, config.API_KEY, config.MODEL, debug=False)
                        else:
                            run_chat(system_prompt, temperature, max_turns)
                        return
                elif choice == 'n':
                    print("\nStarting new pod instead...")
                    break
                else:
                    print("Please enter 'y' or 'n'")

        # Step 1: GPU Selection
        gpu_type, gpu_count, model, script, shell_only, agentic, dual_llm, full_stack = select_gpu_config()

        # Step 1b: Pricing Type Selection
        spot = select_pricing_type()

        # Step 2: Confirmation
        if not confirm_start(gpu_type, gpu_count, model, shell_only, agentic, dual_llm, full_stack, spot):
            print("\nPod start cancelled.")
            sys.exit(0)

        # Step 3: Launch Pod
        pod_id, base_url = launch_pod(gpu_type, gpu_count, model, script, shell_only, agentic, dual_llm, full_stack, spot)

        # Shell-only mode: just show pod info and exit
        if shell_only:
            print("\n=== Shell-Only Pod Ready ===")
            print(f"Pod ID: {pod_id}")
            print(f"SSH: runpodctl connect {pod_id}")
            print(f"Web Terminal: https://{pod_id}-7860.proxy.runpod.net")
            print(f"API Port: https://{pod_id}-8000.proxy.runpod.net")
            print(f"\nMount: /workspace (network volume with vllm, models)")
            print("\nNo auto-start processes. Connect via SSH to begin.")
            print("Use /stop in a new session to terminate when done.")
            input("\nPress Enter to exit...")
            return

        # Agentic mode: show connection info for 7 AI services
        if agentic:
            print("\n=== Agentic Pod Ready ===")
            print(f"Pod ID: {pod_id}")
            print(f"SSH: runpodctl connect {pod_id}")
            print(f"\n7 AI Services Starting:")
            print(f"  [1] Embeddings: https://{pod_id}-8000.proxy.runpod.net")
            print(f"  [2] Reranker:   https://{pod_id}-8001.proxy.runpod.net")
            print(f"  [3] CLIP:       https://{pod_id}-8002.proxy.runpod.net")
            print(f"  [4] BLIP:       https://{pod_id}-8003.proxy.runpod.net")
            print(f"  [5] Memory:     https://{pod_id}-8004.proxy.runpod.net")
            print(f"  [6] TTS:        https://{pod_id}-8005.proxy.runpod.net")
            print(f"  [7] STT:        https://{pod_id}-8006.proxy.runpod.net")
            print(f"\nMount: /workspace (network volume)")
            print("\nServices are starting in background. Check logs via SSH.")
            print("Use /stop in a new session to terminate when done.")
            input("\nPress Enter to exit...")
            return

        # Dual LLM mode: show both endpoints, wait for main model, then chat
        if dual_llm:
            print("\n=== Dual LLM Pod ===")
            print(f"Pod ID: {pod_id}")
            print(f"SSH: runpodctl connect {pod_id}")
            print(f"\nLLM Endpoints:")
            print(f"  [1] Mistral-7B:     https://{pod_id}-8000.proxy.runpod.net")
            print(f"  [2] Stable-Code-3B: https://{pod_id}-8001.proxy.runpod.net")
            print(f"\nMount: /workspace (network volume)")
            print("\nWaiting for Mistral-7B (primary) to be ready...")

        # Full Stack mode: show all 9 endpoints
        if full_stack:
            print("\n=== Full Stack Pod ===")
            print(f"Pod ID: {pod_id}")
            print(f"SSH: runpodctl connect {pod_id}")
            print(f"\nAll 9 Services (starting over ~14 mins):")
            print(f"  [1] Mistral-7B:     https://{pod_id}-8000.proxy.runpod.net")
            print(f"  [2] Stable-Code-3B: https://{pod_id}-8001.proxy.runpod.net")
            print(f"  [3] Reranking:      https://{pod_id}-8002.proxy.runpod.net")
            print(f"  [4] CLIP:           https://{pod_id}-8003.proxy.runpod.net")
            print(f"  [5] BLIP:           https://{pod_id}-8004.proxy.runpod.net")
            print(f"  [6] Memory:         https://{pod_id}-8005.proxy.runpod.net")
            print(f"  [7] TTS:            https://{pod_id}-8006.proxy.runpod.net")
            print(f"  [8] STT:            https://{pod_id}-8007.proxy.runpod.net")
            print(f"  [9] Embedding:      https://{pod_id}-8008.proxy.runpod.net")
            print(f"\nMount: /workspace (network volume)")
            print("\nWaiting for Mistral-7B (primary) to be ready...")

        # Step 4: Wait for vLLM API to be fully ready
        wait_for_vllm_ready(pod_id)

        # Step 5: Chat Configuration
        system_prompt, temperature = select_chat_config()
        max_turns = select_memory_style()
        chat_mode = select_chat_mode()

        # Step 6: Ready to Chat Prompt
        print("\n=== Pod Ready ===")
        while True:
            choice = input("Ready to chat? (y/n): ").strip().lower()
            if choice == 'y':
                print()  # Blank line before chat starts
                if chat_mode == "agent":
                    from my_ai_package import config
                    reload_config_preserve_llm()
                    run_agent(config.LLM_BASE_URL, config.API_KEY, config.MODEL, debug=False)
                else:
                    run_chat(system_prompt, temperature, max_turns)
                break
            elif choice == 'n':
                sys.exit(0)
            else:
                print("Please enter 'y' or 'n'")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
