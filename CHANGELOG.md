# Changelog

All notable changes to SolvX QuickPod will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Server-side idle watchdog - auto-terminates pod after 15 minutes of no connections
- Proper Windows console handler using SetConsoleCtrlHandler API

### Changed
- Loading time message updated to "8-9 minutes"

### Fixed
- Terminal window close not triggering pod termination on Windows

---

## [1.0.0] - 2026-01-28

### Added
- One-click AI chat on RunPod cloud GPUs
- First-run onboarding with guided RunPod account setup
- Automatic model download from HuggingFace (Mistral-7B-Instruct AWQ)
- Streaming chat responses with Rich formatting
- Session history persistence (~/.myai/chat_logs/)
- Pod reconnection support for existing sessions
- Debug mode (`/json`) to view raw API exchanges
- Exit protection - prompts to terminate pod on Ctrl+C to prevent surprise charges
- Emergency pod termination - automatically stops pod if terminal window is closed
- Cost transparency - displays hourly rate at session start
- Desktop shortcut creation (Windows)
- Multi-platform builds (Windows, Linux, macOS)
- Friendly onboarding - helpful guidance if API key not copied, keyboard-mash connection key

### Technical
- vLLM server with AWQ quantization for efficient inference
- RTX 3090 GPU configuration (~$0.44/hour)
- OpenAI-compatible API endpoint
- Conversation history trimming (last 10 turns)

[Unreleased]: https://github.com/tradewithmeai/solvx-quickpod/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/tradewithmeai/solvx-quickpod/releases/tag/v1.0.0
