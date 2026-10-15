"""CPU stand-ins for TRL's Triton kernels — for the smoke test only.

TRL 1.15 routes SFT and GRPO loss computation through a Triton kernel
(`trl.kernels.ChunkedLogProbFunction`) with no CPU fallback, so its trainers
cannot take a step without a GPU. `scripts/smoke_train.py` exercises Mirage's
training code on a CPU, so it swaps in this plain-PyTorch reference.

It is activated only when `MIRAGE_CPU_KERNEL_SHIM=1` is set *and* CUDA is not
available. On Kaggle (a GPU) it never activates and TRL's own kernels run.
"""

from __future__ import annotations

import os


class _ReferenceChunkedLogProb:
    """Same call signature and outputs as trl.kernels.ChunkedLogProbFunction.apply,
    computed by materialising the logits (fine for a tiny model)."""

    @staticmethod
    def apply(hidden, weight, bias, targets, temperature, chunk_size, final_logit_softcapping=None, logit_scale=1.0,
              outputs=("entropy", "log_sum_sq_probs", "mean_logits", "is_top1")):
        import torch

        z = hidden @ weight.t()
        if bias is not None:
            z = z + bias
        z = z * logit_scale
        if final_logit_softcapping is not None:
            z = torch.tanh(z / final_logit_softcapping) * final_logit_softcapping
        z = z.float() / temperature
        logp = torch.log_softmax(z, dim=-1)
        lp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        p = logp.exp()
        ent = -(p * logp).sum(-1) if "entropy" in outputs else None
        lssp = torch.logsumexp(2 * logp, dim=-1) if "log_sum_sq_probs" in outputs else None
        mean = z.mean(-1) if "mean_logits" in outputs else None
        top1 = (z.argmax(-1) == targets) if "is_top1" in outputs else None
        return lp, ent, lssp, mean, top1


def maybe_install() -> bool:
    if os.environ.get("MIRAGE_CPU_KERNEL_SHIM") != "1":
        return False
    import torch

    if torch.cuda.is_available():
        return False
    import trl.trainer.utils as u

    u._ChunkedLogProbFunction = _ReferenceChunkedLogProb
    u._fused_logprob_entropy = None
    print("[mirage] CPU smoke mode: TRL's Triton kernels replaced by a PyTorch reference")
    return True
