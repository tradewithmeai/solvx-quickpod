from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())  # Find and load .env file (searches up directory tree)

import os
import time
import json
import requests
from pathlib import Path

RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
NETWORK_VOLUME_ID = "5ue82zuz7o"
RUNPOD_API = "https://rest.runpod.io/v1/pods"

HEADERS = {
    "Authorization": f"Bearer {RUNPOD_API_KEY}",
    "Content-Type": "application/json",
}

STATE_DIR = Path.home() / ".myai"
STATE_FILE = STATE_DIR / "pod.json"


def start_pod(gpu_type, gpu_count):
    pod_id = create_pod(gpu_type, gpu_count)
    if not pod_id:
        return None

    write_state(pod_id)
    wait_for_running(pod_id)

    base_url = f"https://{pod_id}-8000.proxy.runpod.net"
    wait_for_proxy(base_url)

    return {
        "pod_id": pod_id,
        "base_url": base_url,
    }


def create_pod(gpu_type, gpu_count, script="start_vllm.sh", shell_only=False, agentic=False, dual_llm=False, full_stack=False, spot=False):
    """
    Create a RunPod pod.

    Args:
        gpu_type: GPU type ID string
        gpu_count: Number of GPUs
        script: Startup script name
        shell_only: If True, no auto-start (bare environment)
        agentic: If True, start 7 AI services (embeddings, reranker, clip, blip, memory, tts, stt)
        dual_llm: If True, start Mistral-7B on port 8000 and Stable-Code-3B on port 8001
        full_stack: If True, start both LLMs + all 7 AI services
        spot: If True, use interruptible/spot instance (cheaper but can be interrupted)
    """
    # Cloud type: SECURE (RunPod managed), COMMUNITY (community GPUs)
    # For spot, we use COMMUNITY which typically has spot availability
    cloud_type = "COMMUNITY" if spot else "SECURE"

    if shell_only:
        # Bare CUDA Python pod - no auto-start, mount at /workspace for existing vllm install
        payload = {
            "cloudType": cloud_type,
            "interruptible": spot,
            "computeType": "GPU",
            "gpuCount": gpu_count,
            "gpuTypeIds": [gpu_type],
            "containerDiskInGb": 40,
            "imageName": "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel",
            "supportPublicIp": True,
            "networkVolumeId": NETWORK_VOLUME_ID,
            "volumeMountPath": "/workspace",
            "ports": ["7000/http", "7001/http", "8000/http", "8001/http", "8002/http", "8003/http", "8004/http", "8005/http", "8006/http", "8007/http", "8008/http"],
            "env": {
                "PYTHONPATH": "/workspace/python_pkgs",
                "HF_HOME": "/workspace/hf",
                "TRANSFORMERS_CACHE": "/workspace/hf",
            },
            # Install dependencies, then idle (no server auto-start)
            "dockerStartCmd": [
                "/bin/bash",
                "-c",
                "python3 -m pip install --upgrade pip && "
                "apt-get update && apt-get install -y ffmpeg && "
                "python3 -m pip install 'huggingface-hub<1.0,>=0.34.0' && "
                "python3 -m pip install 'huggingface_hub[cli]' && "
                "sleep infinity",
            ],
        }
    elif agentic:
        # Agentic pod - RTX 3090 with 7 AI services auto-started
        # Scripts block/stay running, so we background them with &
        startup_cmd = (
            "echo '=== Starting Agentic Services ===' && "
            "/workspace/start_embeddings_server.sh & sleep 2 && echo '[1/7] Embeddings started' && "
            "/workspace/reranker/start_api.sh & sleep 2 && echo '[2/7] Reranker started' && "
            "/workspace/clip/start_api.sh & sleep 2 && echo '[3/7] CLIP started' && "
            "/workspace/blip/start_api.sh & sleep 2 && echo '[4/7] BLIP started' && "
            "/workspace/memory/start_api.sh & sleep 2 && echo '[5/7] Memory started' && "
            "/workspace/tts/start_api.sh & sleep 2 && echo '[6/7] TTS started' && "
            "/workspace/start_stt.sh & sleep 2 && echo '[7/7] STT started' && "
            "echo '=== All 7 services running ===' && "
            "sleep infinity"
        )
        payload = {
            "cloudType": cloud_type,
            "interruptible": spot,
            "computeType": "GPU",
            "gpuCount": gpu_count,
            "gpuTypeIds": [gpu_type],
            "containerDiskInGb": 40,
            "imageName": "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel",
            "supportPublicIp": True,
            "networkVolumeId": NETWORK_VOLUME_ID,
            "volumeMountPath": "/workspace",
            "ports": ["7000/http", "7001/http", "8000/http", "8001/http", "8002/http", "8003/http", "8004/http", "8005/http", "8006/http", "8007/http", "8008/http"],
            "env": {
                "PYTHONPATH": "/workspace/python_pkgs",
                "HF_HOME": "/workspace/hf",
                "TRANSFORMERS_CACHE": "/workspace/hf",
                "COQUI_TOS_AGREED": "1",
            },
            "dockerStartCmd": [
                "/bin/bash",
                "-c",
                startup_cmd,
            ],
        }
    elif dual_llm:
        # Dual LLM pod - Mistral-7B on port 8000, Stable-Code-3B on port 8001
        api_key = os.getenv("VLLM_API_KEY", "rk_PLACEHOLDER")

        # Mistral chat template - base64 encoded to avoid shell escaping issues
        import base64
        mistral_template_b64 = base64.b64encode(
            b"{{ bos_token }}{% for message in messages %}{% if message['role'] == 'user' %}[INST] {{ message['content'] }} [/INST]{% elif message['role'] == 'assistant' %}{{ message['content'] }}{{ eos_token }}{% endif %}{% endfor %}"
        ).decode()

        startup_cmd = (
            "echo '=== Starting Dual LLM Setup ===' && "
            "pip install --upgrade pip && "
            "python3 -c 'import vllm' 2>/dev/null || (echo 'Installing vLLM...' && pip install vllm && echo 'vLLM installed') && "
            "mkdir -p /workspace/chat_templates && "
            f"echo '{mistral_template_b64}' | base64 -d > /workspace/chat_templates/mistral.jinja && "
            "echo '[1/2] Starting Mistral-7B on port 8000...' && "
            f"python3 -m vllm.entrypoints.openai.api_server "
            f"--model /workspace/models/mistral-7b-instruct-awq "
            f"--gpu-memory-utilization 0.7 "
            f"--tensor-parallel-size 1 "
            f"--host 0.0.0.0 "
            f"--port 8000 "
            f"--api-key {api_key} "
            f"--chat-template /workspace/chat_templates/mistral.jinja > /workspace/vllm_mistral.log 2>&1 & "
            "echo 'Waiting 7 mins for Mistral to fully load...' && "
            "sleep 420 && "
            "echo '[2/2] Starting Stable-Code-3B on port 8001...' && "
            f"python3 -m vllm.entrypoints.openai.api_server "
            f"--model /workspace/models/stable-code-instruct-3b-awq "
            f"--gpu-memory-utilization 0.23 "
            f"--max-model-len 512 "
            f"--max-num-seqs 4 "
            f"--tensor-parallel-size 1 "
            f"--host 0.0.0.0 "
            f"--port 8001 "
            f"--api-key {api_key} > /workspace/vllm_stable.log 2>&1 & "
            "echo '=== Both LLM servers starting ===' && "
            "echo 'Logs: /workspace/vllm_mistral.log and /workspace/vllm_stable.log' && "
            "sleep infinity"
        )
        payload = {
            "cloudType": cloud_type,
            "interruptible": spot,
            "computeType": "GPU",
            "gpuCount": gpu_count,
            "gpuTypeIds": [gpu_type],
            "containerDiskInGb": 40,
            "imageName": "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel",
            "supportPublicIp": True,
            "networkVolumeId": NETWORK_VOLUME_ID,
            "volumeMountPath": "/workspace",
            "ports": ["7000/http", "7001/http", "8000/http", "8001/http", "8002/http", "8003/http", "8004/http", "8005/http", "8006/http", "8007/http", "8008/http"],
            "env": {
                "TENSOR_PARALLEL_SIZE": "1",
                "VLLM_API_KEY": api_key,
                "HF_HOME": "/workspace/hf",
                "TRANSFORMERS_CACHE": "/workspace/hf",
            },
            "dockerStartCmd": [
                "/bin/bash",
                "-c",
                startup_cmd,
            ],
        }
    elif full_stack:
        # Full Stack pod - both LLMs + 7 AI services
        # Strategy: Start CPU services first (lightweight), then LLMs after 5 min delay
        api_key = os.getenv("VLLM_API_KEY", "rk_PLACEHOLDER")

        # Mistral chat template - base64 encoded to avoid shell escaping issues
        import base64
        mistral_template_b64 = base64.b64encode(
            b"{{ bos_token }}{% for message in messages %}{% if message['role'] == 'user' %}[INST] {{ message['content'] }} [/INST]{% elif message['role'] == 'assistant' %}{{ message['content'] }}{{ eos_token }}{% endif %}{% endfor %}"
        ).decode()

        startup_cmd = (
            # PHASE 1: Start all CPU services, then wait 5 mins
            "echo '=== PHASE 1: Starting CPU Services ===' && "
            "/workspace/reranker/start_api.sh & "
            "sleep 2 && "
            "/workspace/clip/start_api.sh & "
            "sleep 2 && "
            "/workspace/blip/start_api.sh & "
            "sleep 2 && "
            "/workspace/memory/start_api.sh & "
            "sleep 2 && "
            "/workspace/tts/start_api.sh & "
            "sleep 2 && "
            "/workspace/start_stt.sh & "
            "sleep 2 && "
            "/workspace/start_embeddings_server.sh & "
            "sleep 2 && "
            "echo 'CPU services started. Waiting 5 mins...' && "
            "sleep 300 && "

            # PHASE 2: Start Mistral vLLM (same method as standard pod), then wait 10 mins
            "echo '=== PHASE 2: Starting Mistral-7B ===' && "
            "unset PYTHONPATH && "
            "pip install --upgrade pip && "
            "python3 -c 'import vllm' 2>/dev/null || (echo 'Installing vLLM...' && pip install vllm && echo 'vLLM installed') && "
            "mkdir -p /workspace/chat_templates && "
            f"echo '{mistral_template_b64}' | base64 -d > /workspace/chat_templates/mistral.jinja && "
            f"python3 -m vllm.entrypoints.openai.api_server "
            f"--model /workspace/models/mistral-7b-instruct-awq "
            f"--gpu-memory-utilization 0.7 "
            f"--tensor-parallel-size 1 "
            f"--host 0.0.0.0 "
            f"--port 8000 "
            f"--api-key {api_key} "
            f"--chat-template /workspace/chat_templates/mistral.jinja > /workspace/vllm_mistral.log 2>&1 & "
            "echo 'Mistral loading. Waiting 15 mins before Phase 3...' && "
            "sleep 900 && "

            # PHASE 3: Start Stable-Code vLLM
            "echo '=== PHASE 3: Starting Stable-Code-3B ===' && "
            "unset PYTHONPATH && "
            f"python3 -m vllm.entrypoints.openai.api_server "
            f"--model /workspace/models/stable-code-instruct-3b-awq "
            f"--gpu-memory-utilization 0.23 "
            f"--max-model-len 512 "
            f"--max-num-seqs 4 "
            f"--tensor-parallel-size 1 "
            f"--host 0.0.0.0 "
            f"--port 8001 "
            f"--api-key {api_key} > /workspace/vllm_stable.log 2>&1 & "
            "echo '=== ALL 9 SERVICES STARTED ===' && "
            "echo 'Ports: CPU 8002-8008 | Mistral 8000 | Stable-Code 8001' && "
            "sleep infinity"
        )
        payload = {
            "cloudType": cloud_type,
            "interruptible": spot,
            "computeType": "GPU",
            "gpuCount": gpu_count,
            "gpuTypeIds": [gpu_type],
            "containerDiskInGb": 40,
            "imageName": "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel",
            "supportPublicIp": True,
            "networkVolumeId": NETWORK_VOLUME_ID,
            "volumeMountPath": "/workspace",
            "ports": ["7000/http", "7001/http", "8000/http", "8001/http", "8002/http", "8003/http", "8004/http", "8005/http", "8006/http", "8007/http", "8008/http"],
            "env": {
                "TENSOR_PARALLEL_SIZE": "1",
                "VLLM_API_KEY": api_key,
                "HF_HOME": "/workspace/hf",
                "TRANSFORMERS_CACHE": "/workspace/hf",
                "PYTHONPATH": "/workspace/python_pkgs",
                "COQUI_TOS_AGREED": "1",
            },
            "dockerStartCmd": [
                "/bin/bash",
                "-c",
                startup_cmd,
            ],
        }
    else:
        # Standard vLLM pod - install vLLM if needed, then start server
        # Get model path from script name (default to mistral)
        if "qwen32b" in script:
            model_path = "/workspace/models/qwen2.5-32b-awq"
            gpu_mem = "0.9"
        else:
            model_path = "/workspace/models/mistral-7b-instruct-awq"
            gpu_mem = "0.7"

        api_key = os.getenv("VLLM_API_KEY", "rk_PLACEHOLDER")

        # Mistral chat template - base64 encoded to avoid shell escaping issues
        # Original: {{ bos_token }}{% for message in messages %}{% if message['role'] == 'user' %}[INST] {{ message['content'] }} [/INST]{% elif message['role'] == 'assistant' %}{{ message['content'] }}{{ eos_token }}{% endif %}{% endfor %}
        import base64
        mistral_template_b64 = base64.b64encode(
            b"{{ bos_token }}{% for message in messages %}{% if message['role'] == 'user' %}[INST] {{ message['content'] }} [/INST]{% elif message['role'] == 'assistant' %}{{ message['content'] }}{{ eos_token }}{% endif %}{% endfor %}"
        ).decode()

        # Startup command that installs vLLM if missing, creates chat template, then runs server
        startup_cmd = (
            f"echo '=== Starting vLLM Setup ===' && "
            f"pip install --upgrade pip && "
            f"python3 -c 'import vllm' 2>/dev/null || (echo 'Installing vLLM...' && pip install vllm && echo 'vLLM installed successfully') && "
            f"mkdir -p /workspace/chat_templates && "
            f"echo '{mistral_template_b64}' | base64 -d > /workspace/chat_templates/mistral.jinja && "
            f"echo 'Starting vLLM server...' && "
            f"python3 -m vllm.entrypoints.openai.api_server "
            f"--model {model_path} "
            f"--gpu-memory-utilization {gpu_mem} "
            f"--tensor-parallel-size {gpu_count} "
            f"--host 0.0.0.0 "
            f"--port 8000 "
            f"--api-key {api_key} "
            f"--chat-template /workspace/chat_templates/mistral.jinja"
        )

        payload = {
            "cloudType": cloud_type,
            "interruptible": spot,
            "computeType": "GPU",
            "gpuCount": gpu_count,
            "gpuTypeIds": [gpu_type],
            "containerDiskInGb": 40,
            "imageName": "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel",
            "supportPublicIp": True,
            "networkVolumeId": NETWORK_VOLUME_ID,
            "volumeMountPath": "/workspace",
            "ports": ["7000/http", "7001/http", "8000/http", "8001/http", "8002/http", "8003/http", "8004/http", "8005/http", "8006/http", "8007/http", "8008/http"],
            "env": {
                "TENSOR_PARALLEL_SIZE": str(gpu_count),
                "VLLM_API_KEY": api_key,
                "HF_HOME": "/workspace/hf",
                "TRANSFORMERS_CACHE": "/workspace/hf",
            },
            "dockerStartCmd": [
                "/bin/bash",
                "-c",
                startup_cmd,
            ],
        }

    r = requests.post(RUNPOD_API, headers=HEADERS, json=payload, timeout=30)
    if r.status_code not in (200, 201):
        print("Create failed:", r.text)
        return None

    pod_id = r.json()["id"]
    print(f"Pod created: {pod_id}")
    return pod_id


def wait_for_running(pod_id):
    print("Waiting for pod to be ready...")
    last_seen = None

    while True:
        r = requests.get(f"{RUNPOD_API}/{pod_id}", headers=HEADERS, timeout=15)
        r.raise_for_status()
        data = r.json()

        # Use correct RunPod API field names
        desired = data.get("desiredStatus")
        last_started = data.get("lastStartedAt")
        machine_id = data.get("machineId")
        public_ip = data.get("publicIp")

        # Show status changes for debugging
        snapshot = (desired, bool(last_started), machine_id, public_ip)
        if snapshot != last_seen:
            print(
                f"  desiredStatus={desired} "
                f"lastStartedAt={'set' if last_started else 'unset'} "
                f"machineId={machine_id} "
                f"publicIp={'set' if public_ip else 'unset'}",
                flush=True,
            )
            last_seen = snapshot

        # Pod is ready when desiredStatus is RUNNING and lastStartedAt is set
        if desired == "RUNNING" and last_started:
            return

        time.sleep(3)


def wait_for_proxy(base_url):
    print("Waiting for proxy / port 8000...")
    while True:
        try:
            r = requests.get(base_url, timeout=5)
            if r.status_code < 500:
                print("Proxy is live.")
                return
        except Exception:
            pass
        time.sleep(3)


def write_state(pod_id, model=None):
    STATE_DIR.mkdir(exist_ok=True)
    state = {"pod_id": pod_id}
    if model:
        state["model"] = model
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)


def clear_state():
    """Clear the state file when pod is no longer valid."""
    if STATE_FILE.exists():
        STATE_FILE.unlink()
        print("[State cleared - old pod no longer running]")


def is_pod_running(pod_id):
    """Check if pod is still running. Returns True if running, False if terminated."""
    try:
        r = requests.get(f"{RUNPOD_API}/{pod_id}", headers=HEADERS, timeout=10)
        if r.status_code == 404:
            return False  # Pod doesn't exist
        r.raise_for_status()
        data = r.json()
        desired = data.get("desiredStatus")
        # Pod is running if desiredStatus is RUNNING
        return desired == "RUNNING"
    except requests.RequestException:
        # Network error - assume pod might still be running
        return True


def terminate_pod(pod_id):
    """Terminate a running pod. Returns True if successful."""
    try:
        r = requests.delete(f"{RUNPOD_API}/{pod_id}", headers=HEADERS, timeout=30)
        # Success if deleted or already gone
        return r.status_code in (200, 204, 404)
    except requests.RequestException:
        return False
