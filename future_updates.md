# Future Updates

## Completed

- ~~Pod Termination Detection~~ - Time-based checks every 10 seconds
- ~~Multi-GPU Support (Tensor Parallel)~~ - Passes TENSOR_PARALLEL_SIZE to vLLM
- ~~Session Recovery / Re-entry~~ - Detects existing pod and offers reconnect
- ~~Terminate Pod from Terminal~~ - `/stop` command in chat
- ~~Memory Options~~ - Sliding window vs full history
- ~~API Key Fix~~ - Passes VLLM_API_KEY to pod environment

## Remaining

### Additional Models
- Configure RTX 4090 setup
- Add full precision Qwen 2.5-32B option (currently using AWQ quantized)

### Reasoning/Planning Loops
- Add optional "thinking" phase before generating response
- Presets: plan, critique, step-by-step
- Deferred for later implementation

### Judge Model
- Add second model to evaluate/critique responses
- Future feature for research/comparison

### Playground Mode (Long-term)
- Custom system prompt editor (free text)
- Adjustable max_tokens
- View/edit message history
- Export conversations
- Token counting display
- Compare outputs across settings
