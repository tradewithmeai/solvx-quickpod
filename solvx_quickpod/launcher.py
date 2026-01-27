from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())  # Find and load .env file (searches up directory tree)

import os
import time
import json
import requests
from pathlib import Path

RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
RUNPOD_API = "https://rest.runpod.io/v1/pods"

# HuggingFace model to download
HF_MODEL_ID = "TheBloke/Mistral-7B-Instruct-v0.2-AWQ"

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


def create_pod(gpu_type, gpu_count):
    """
    Create a RunPod pod with Mistral-7B vLLM server.
    Downloads model from HuggingFace on first launch.

    Args:
        gpu_type: GPU type ID string
        gpu_count: Number of GPUs
    """
    gpu_mem = "0.9"
    api_key = os.getenv("VLLM_API_KEY", "rk_PLACEHOLDER")

    # Startup command:
    # 1. Install vLLM
    # 2. Download model from HuggingFace (cached in HF_HOME)
    # 3. Start vLLM server with AWQ quantization
    startup_cmd = (
        f"echo '=== Starting vLLM Setup ===' && "
        f"pip install -q --upgrade pip && "
        f"pip install -q vllm && "
        f"echo 'Downloading model from HuggingFace (first run takes a few minutes)...' && "
        f"python3 -m vllm.entrypoints.openai.api_server "
        f"--model {HF_MODEL_ID} "
        f"--quantization awq "
        f"--dtype auto "
        f"--gpu-memory-utilization {gpu_mem} "
        f"--tensor-parallel-size {gpu_count} "
        f"--host 0.0.0.0 "
        f"--port 8000 "
        f"--api-key {api_key}"
    )

    payload = {
        "cloudType": "SECURE",
        "interruptible": False,
        "computeType": "GPU",
        "gpuCount": gpu_count,
        "gpuTypeIds": [gpu_type],
        "containerDiskInGb": 50,  # Extra space for model download (~4GB) + vLLM + cache
        "imageName": "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel",
        "supportPublicIp": True,
        "ports": ["8000/http"],
        "env": {
            "TENSOR_PARALLEL_SIZE": str(gpu_count),
            "VLLM_API_KEY": api_key,
            "HF_HOME": "/root/.cache/huggingface",
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
