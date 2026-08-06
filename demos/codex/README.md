Use `chat_template_qwen36_codex.jinja` - it has support for developer role

Lookup `config.toml` because it has disabled functions with for some reason do not work with OVMS as of today

Edit config.toml to select correct model and point to correct service

Build codex-rs using these instructions: https://github.com/openai/codex/blob/main/docs/install.md

Run built codex-rs with this command (full yolo mode):
```
cargo run --bin codex -- --yolo -c 'model_provider="local_server"' -c 'model="ovms-model"'
```


Limitations:
- instruction field of responses API is completely ignored

