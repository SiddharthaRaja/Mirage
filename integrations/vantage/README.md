# Mirage × Vantage: cost per correct query

Vantage measures serving cost and latency; Mirage measures correctness. This
folder joins them.

## 1. Patch Vantage (once)

Vantage builds RAG-shaped prompts by default. To benchmark exactly the prompt
the SQL model was trained on, apply the patch from the Vantage repo root:

```bash
git apply ../mirage/integrations/vantage/vantage-raw-prompt.patch
pytest -q        # Vantage's own suite still passes with it (59 tests)
```

It adds two backwards-compatible things: workload rows may carry a verbatim
`prompt`, and a sweep may set `system:`.

## 2. Export the workload

```bash
mirage vantage export -c configs/grpo_base.yaml --split dev --out ../vantage/data/workload.mirage-dev.jsonl
```

## 3. Serve all three models from one process

```bash
vllm serve runs/sft/adapter-merged --port 8000 --dtype half --max-model-len 4096 \
  --enable-lora --max-lora-rank 16 \
  --lora-modules mirage-single=runs/arms/single/adapter mirage-suite=runs/arms/suite/adapter
```

One process means one GPU, one scheduler, one set of kernels: the only
difference between the three measurements is the model. (If you skipped SFT,
serve the base model name instead of the merged folder.)

## 4. Run the sweep

Copy `backends/*.yaml` into `vantage/configs/backends/` and
`sweep.mirage-cost.yaml` into `vantage/configs/sweeps/`, set `gpu_hourly_usd`
to what you actually pay, then:

```bash
vantage sweep -c configs/sweeps/sweep.mirage-cost.yaml
```

Run the load generator on a different machine from the GPU if you can.

## 5. Join

```bash
mirage vantage cost <vantage run dir> \
  --pair "start=base[concurrency=8]:runs/eval/start-dev" \
  --pair "grpo-single=grpo-single[concurrency=8]:runs/eval/grpo-single-dev" \
  --pair "grpo-suite=grpo-suite[concurrency=8]:runs/eval/grpo-suite-dev" \
  --out runs/vantage/cost.json
```

Then add `cost: runs/vantage/cost.json` to the study file.

`sweep.mirage-cost-sim.yaml` is the simulator version that produced the
demo's bundled cost data; it says nothing about real hardware.
