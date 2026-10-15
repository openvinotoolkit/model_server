---
id: ovms_docs_prepare_genai
sidebar_position: 5
---

# Exporting GEN AI Models

:::info Phase 1 structure note (pending stakeholder review)
GenAI model preparation chooser.
:::


Prepare models using OVMS [pull mode](pull-from-huggingface.md) (recommended).

Prepare models using [python script](../../demos/common/export_models/README.md).

Prepare models using OVMS with python [optimum pull mode](export-with-optimum-cli.md).

> **Note:** For several large generative models on limited memory, use [Idle servable management](../deploy-and-operate/idle-servable-management.md) (preview). Configure `--cache_dir` to reduce model wake-up latency.
