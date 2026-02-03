#!/usr/bin/env python3
"""
SolvX QuickPod - Pod Launcher Module

Manages RunPod GPU pod lifecycle including:
- Pod creation with vLLM server configuration
- Pod status monitoring
- Pod termination
- State persistence

The pod runs a vLLM server with the Mistral-7B-Instruct AWQ model,
downloaded automatically from HuggingFace on first launch.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import requests
from dotenv import load_dotenv

# =============================================================================
# CONFIGURATION
# =============================================================================

# Load environment variables from ~/.myai/.env
_ENV_PATH: Path = Path.home() / ".myai" / ".env"
load_dotenv(_ENV_PATH)

# API Configuration
RUNPOD_API_KEY: Optional[str] = os.getenv("RUNPOD_API_KEY")
RUNPOD_API_URL: str = "https://rest.runpod.io/v1/pods"

# Model Configuration
HF_MODEL_ID: str = "TheBloke/Mistral-7B-Instruct-v0.2-AWQ"

# State Persistence
STATE_DIR: Path = Path.home() / ".myai"
STATE_FILE: Path = STATE_DIR / "pod.json"

# API Request Headers
_HEADERS: Dict[str, str] = {
    "Authorization": f"Bearer {RUNPOD_API_KEY}",
    "Content-Type": "application/json",
}


# =============================================================================
# POD LIFECYCLE - HIGH LEVEL
# =============================================================================

def start_pod(gpu_type: str, gpu_count: int) -> Optional[Dict[str, str]]:
    """
    Create and start a new RunPod GPU pod.

    This is the main entry point for pod creation. It handles the complete
    lifecycle: creation, waiting for running state, and proxy availability.

    Args:
        gpu_type: GPU model identifier (e.g., "NVIDIA GeForce RTX 3090")
        gpu_count: Number of GPUs to allocate

    Returns:
        Dictionary with 'pod_id' and 'base_url' on success, None on failure.
    """
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


# =============================================================================
# POD CREATION
# =============================================================================

def create_pod(gpu_type: str, gpu_count: int) -> Optional[str]:
    """
    Create a RunPod pod configured with vLLM and Mistral-7B.

    The pod is configured to:
    1. Install vLLM on startup
    2. Download the model from HuggingFace (cached for pod lifetime)
    3. Start the vLLM OpenAI-compatible API server

    Args:
        gpu_type: GPU model identifier
        gpu_count: Number of GPUs to allocate

    Returns:
        Pod ID string on success, None on failure.
    """
    gpu_memory_utilization = "0.9"
    api_key = os.getenv("VLLM_API_KEY", "default_key")
    runpod_key = os.getenv("RUNPOD_API_KEY", "")

    # Idle watchdog script - auto-terminates pod if no client connects for 15 min.
    # Safety net for: terminal close, client crash, internet loss, power failure.
    # RUNPOD_POD_ID is provided automatically by RunPod inside the container.
    idle_watchdog = (
        "("
        "  IDLE_LIMIT=900; "  # 15 minutes in seconds
        "  LAST_ACTIVE=$(date +%s); "
        "  sleep 600; "  # 10 min grace period for model loading
        "  while true; do "
        "    sleep 120; "  # Check every 2 minutes
        "    CONNS=$(ss -tn state established 'sport = :8000' 2>/dev/null | tail -n +2 | wc -l); "
        "    NOW=$(date +%s); "
        "    if [ \"$CONNS\" -gt \"0\" ]; then "
        "      LAST_ACTIVE=$NOW; "
        "    fi; "
        "    IDLE=$((NOW - LAST_ACTIVE)); "
        "    if [ \"$IDLE\" -gt \"$IDLE_LIMIT\" ]; then "
        "      curl -s -X DELETE \"https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID\" "
        "        -H \"Authorization: Bearer $RUNPOD_TERMINATE_KEY\" "
        "        -H \"Content-Type: application/json\" > /dev/null 2>&1; "
        "      exit 0; "
        "    fi; "
        "  done"
        ") &"
    )

    # Build the startup command
    # 1. Start idle watchdog in background
    # 2. Install vLLM
    # 3. Start the vLLM server with AWQ quantization
    startup_cmd = (
        f"{idle_watchdog} "
        f"echo '=== Starting vLLM Setup ===' && "
        f"pip install -q --upgrade pip && "
        f"pip install -q vllm && "
        f"echo 'Downloading model from HuggingFace...' && "
        f"python3 -m vllm.entrypoints.openai.api_server "
        f"--model {HF_MODEL_ID} "
        f"--quantization awq "
        f"--dtype auto "
        f"--gpu-memory-utilization {gpu_memory_utilization} "
        f"--tensor-parallel-size {gpu_count} "
        f"--host 0.0.0.0 "
        f"--port 8000 "
        f"--api-key {api_key}"
    )

    # Pod configuration payload
    payload: Dict[str, Any] = {
        "cloudType": "SECURE",
        "interruptible": False,
        "computeType": "GPU",
        "gpuCount": gpu_count,
        "gpuTypeIds": [gpu_type],
        "containerDiskInGb": 50,  # Space for model (~4GB) + vLLM + cache
        "imageName": "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel",
        "supportPublicIp": True,
        "ports": ["8000/http"],
        "env": {
            "TENSOR_PARALLEL_SIZE": str(gpu_count),
            "VLLM_API_KEY": api_key,
            "HF_HOME": "/root/.cache/huggingface",
            "RUNPOD_TERMINATE_KEY": runpod_key,
        },
        "dockerStartCmd": ["/bin/bash", "-c", startup_cmd],
    }

    try:
        response = requests.post(
            RUNPOD_API_URL,
            headers=_HEADERS,
            json=payload,
            timeout=30,
        )

        if response.status_code not in (200, 201):
            print(f"Create failed: {response.text}")
            return None

        pod_id = response.json()["id"]
        print(f"Pod created: {pod_id}")
        return pod_id

    except requests.RequestException as e:
        print(f"Create failed: {e}")
        return None


# =============================================================================
# POD STATUS MONITORING
# =============================================================================

def wait_for_running(pod_id: str) -> None:
    """
    Wait for a pod to reach the RUNNING state.

    Polls the RunPod API and displays status updates until the pod
    is fully running with a machine assigned.

    Args:
        pod_id: The pod identifier to monitor.
    """
    print("Starting GPU pod...")
    last_stage: Optional[str] = None

    while True:
        try:
            response = requests.get(
                f"{RUNPOD_API_URL}/{pod_id}",
                headers=_HEADERS,
                timeout=15,
            )
            response.raise_for_status()
            data = response.json()

            # Extract status fields
            desired_status = data.get("desiredStatus")
            last_started = data.get("lastStartedAt")
            machine_id = data.get("machineId")
            public_ip = data.get("publicIp")

            # Determine user-friendly stage
            if machine_id and last_started:
                stage = "Container starting..."
            elif machine_id:
                stage = "GPU assigned, preparing container..."
            elif desired_status == "RUNNING":
                stage = "Finding available GPU..."
            else:
                stage = "Initializing..."

            # Display stage changes
            if stage != last_stage:
                print(f"  {stage}", flush=True)
                last_stage = stage

            # Pod is ready when RUNNING and started
            if desired_status == "RUNNING" and last_started:
                print("  GPU pod is running!")
                return

        except requests.RequestException:
            pass  # Retry on network errors

        time.sleep(3)


def wait_for_proxy(base_url: str) -> None:
    """
    Wait for the pod's HTTP proxy to become available.

    Polls the base URL until it responds with a non-5xx status code,
    indicating the proxy is ready to forward requests.

    Args:
        base_url: The pod's proxy URL to check.
    """
    print("Connecting to server...")

    while True:
        try:
            response = requests.get(base_url, timeout=5)
            if response.status_code < 500:
                print("  Server connection established!")
                return
        except requests.RequestException:
            pass  # Retry on connection errors

        time.sleep(3)


def is_pod_running(pod_id: str) -> bool:
    """
    Check if a pod is currently running.

    Args:
        pod_id: The pod identifier to check.

    Returns:
        True if the pod is running, False if terminated or not found.
    """
    try:
        response = requests.get(
            f"{RUNPOD_API_URL}/{pod_id}",
            headers=_HEADERS,
            timeout=10,
        )

        if response.status_code == 404:
            return False  # Pod doesn't exist

        response.raise_for_status()
        data = response.json()

        return data.get("desiredStatus") == "RUNNING"

    except requests.RequestException:
        # Network error - assume pod might still be running
        return True


# =============================================================================
# POD TERMINATION
# =============================================================================

def terminate_pod(pod_id: str) -> bool:
    """
    Terminate a running pod.

    Args:
        pod_id: The pod identifier to terminate.

    Returns:
        True if termination was successful, False otherwise.
    """
    try:
        response = requests.delete(
            f"{RUNPOD_API_URL}/{pod_id}",
            headers=_HEADERS,
            timeout=30,
        )
        # Success if deleted or already gone
        return response.status_code in (200, 204, 404)

    except requests.RequestException:
        return False


# =============================================================================
# STATE PERSISTENCE
# =============================================================================

def write_state(pod_id: str, model: Optional[str] = None) -> None:
    """
    Save pod state to the state file.

    Args:
        pod_id: The active pod identifier.
        model: Optional model identifier to store.
    """
    STATE_DIR.mkdir(exist_ok=True)

    state: Dict[str, str] = {"pod_id": pod_id}
    if model:
        state["model"] = model

    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f)


def clear_state() -> None:
    """
    Clear the pod state file.

    Called when a pod is no longer valid to prevent stale reconnection attempts.
    """
    if STATE_FILE.exists():
        STATE_FILE.unlink()
        print("[State cleared - old pod no longer running]")
