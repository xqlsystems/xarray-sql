"""Check and time native array frameworks on the SQL nanoGPT workload.

Uses the same model, initial weights, five Shakespeare batches and SGD as
``benchmarks/ml/02_nanogpt.py``. JAX (https://github.com/jax-ml/jax) runs
on its installed device. PyTorch (https://pytorch.org) imports the original
nanoGPT model (https://github.com/karpathy/nanoGPT), with matching dimensions
and tanh GELU. CPU uses float64; Apple's MPS GPU uses float32. Each timed
step returns the loss and updated parameters, checked against the SQL
demo's float64 JAX reference outside the timer. Device synchronization is
included. Source downloads and package installation are separate.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

try:
    import torch
except ImportError:
    torch = None

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "benchmarks/ml"))


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _torch_load(model, params):
    with torch.no_grad():
        for name, target in (
            ("wte", model.transformer.wte.weight),
            ("wpe", model.transformer.wpe.weight),
            ("ln_f", model.transformer.ln_f.weight),
        ):
            target.copy_(
                torch.as_tensor(
                    params[name], dtype=target.dtype, device=target.device
                )
            )
        for layer, block in enumerate(model.transformer.h):
            packed = np.concatenate(
                [
                    params[name][layer].reshape(768, 768).T
                    for name in ("wq", "wk", "wv")
                ]
            )
            for target, source in (
                (block.attn.c_attn.weight, packed),
                (
                    block.attn.c_proj.weight,
                    params["wo"][layer].reshape(768, 768).T,
                ),
                (block.mlp.c_fc.weight, params["w_fc"][layer].T),
                (block.mlp.c_proj.weight, params["w_proj"][layer].T),
                (block.ln_1.weight, params["ln_1"][layer]),
                (block.ln_2.weight, params["ln_2"][layer]),
            ):
                target.copy_(
                    torch.as_tensor(
                        source, dtype=target.dtype, device=target.device
                    )
                )
            # The SQL demo uses GPT-2's tanh approximation; upstream's
            # default nn.GELU uses erf. Keep the benchmark math identical.
            block.mlp.gelu = torch.nn.GELU(approximate="tanh")


def _torch_arrays(model):
    def array(tensor):
        return tensor.detach().cpu().numpy()

    yield "wte", array(model.transformer.wte.weight)
    yield "wpe", array(model.transformer.wpe.weight)
    yield "ln_f", array(model.transformer.ln_f.weight)
    for name in ("wq", "wk", "wv", "wo", "w_fc", "w_proj", "ln_1", "ln_2"):
        layers = []
        for block in model.transformer.h:
            if name in ("wq", "wk", "wv"):
                index = ("wq", "wk", "wv").index(name)
                layers.append(
                    array(
                        block.attn.c_attn.weight[
                            index * 768 : (index + 1) * 768
                        ]
                    ).T.reshape(768, 12, 64)
                )
            elif name == "wo":
                layers.append(
                    array(block.attn.c_proj.weight).T.reshape(12, 64, 768)
                )
            elif name == "w_fc":
                layers.append(array(block.mlp.c_fc.weight).T)
            elif name == "w_proj":
                layers.append(array(block.mlp.c_proj.weight).T)
            else:
                layers.append(array(getattr(block, name).weight))
        yield name, np.stack(layers)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        required=True,
        choices=[
            "jax_eager",
            "jax_separate",
            "jax_jit",
            "torch_cpu",
            "torch_mps",
        ],
    )
    parser.add_argument("--context", type=int, default=16)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--forward-only", action="store_true")
    parser.add_argument("--nanogpt-source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.context, args.steps) <= 0:
        parser.error("context and steps must be positive")
    n = _module("xql_nanogpt_reference", _ROOT / "benchmarks/ml/02_nanogpt.py")
    n.LAYERS, n.EMBED, n.HEADS, n.HEAD, n.HIDDEN, n.CONTEXT = (
        12,
        768,
        12,
        64,
        3072,
        args.context,
    )
    init = n.init_params(50257, np.random.default_rng(0))
    data = list(
        zip(
            range(args.steps),
            n.batches(n.shakespeare(), 1, np.random.default_rng(1)),
        )
    )
    reference = dict(init)
    value_grad = jax.value_and_grad(n.jax_loss)
    lr = 0.01
    gpu = args.backend == "torch_mps"
    tolerance = 3e-5 if gpu else 1e-9
    result = {
        "backend": args.backend,
        "context": args.context,
        "parameters": sum(w.size for w in init.values()),
        "parameter_shapes": {name: list(w.shape) for name, w in init.items()},
        "parameter_counts": {name: w.size for name, w in init.items()},
        "steps": args.steps,
        "batch": 1,
        "learning_rate": lr,
        "optimizer": "sgd",
        "gelu": "tanh",
        "forward_only": args.forward_only,
        "rtol": tolerance,
        "atol": tolerance,
        "python": platform.python_version(),
        "jax_version": jax.__version__,
        "reference_sha256": hashlib.sha256(
            (_ROOT / "benchmarks/ml/02_nanogpt.py").read_bytes()
        ).hexdigest(),
        "rows": [],
        "compile_seconds": 0.0,
        "scope": "One loss forward and, for training, its backward and SGD update; synchronization included; validation and input/state setup excluded.",
    }
    started = time.perf_counter()
    if args.backend.startswith("jax"):
        params = jax.tree.map(lambda w: jnp.array(w, copy=True), init)
        jax.block_until_ready(params)
        result["devices"] = [str(d) for d in jax.devices()]

        def train_step(p, tok, nxt):
            if args.backend == "jax_separate":
                loss = n.jax_loss(p, tok, nxt)
                jax.block_until_ready(loss)
                gradients = jax.grad(n.jax_loss)(p, tok, nxt)
            else:
                loss, gradients = value_grad(p, tok, nxt)
            return {
                name: w - lr * gradients[name] for name, w in p.items()
            }, loss

        function = n.jax_loss if args.forward_only else train_step
        if args.backend == "jax_jit":
            begin = time.perf_counter()
            tok, nxt = data[0][1]["tok"].values, data[0][1]["next"].values
            function = jax.jit(
                function, donate_argnums=() if args.forward_only else (0,)
            )
            function = function.lower(params, tok, nxt).compile()
            result["compile_seconds"] = time.perf_counter() - begin
    else:
        if torch is None or args.nanogpt_source is None:
            parser.error(
                "PyTorch and --nanogpt-source are required for torch backends"
            )
        torch.set_num_threads(30)
        if gpu and not torch.backends.mps.is_available():
            parser.error("MPS GPU is unavailable")
        upstream = _module(
            "xql_upstream_nanogpt", args.nanogpt_source / "model.py"
        )
        result["nanogpt_revision"] = subprocess.check_output(
            ["git", "-C", str(args.nanogpt_source), "rev-parse", "HEAD"],
            text=True,
        ).strip()
        result["nanogpt_model_sha256"] = hashlib.sha256(
            (args.nanogpt_source / "model.py").read_bytes()
        ).hexdigest()
        result["torch_version"] = torch.__version__
        device, dtype = (
            ("mps", torch.float32) if gpu else ("cpu", torch.float64)
        )
        old = torch.get_default_dtype()
        try:
            torch.set_default_dtype(dtype)
            model = upstream.GPT(
                upstream.GPTConfig(
                    block_size=args.context,
                    vocab_size=50257,
                    n_layer=12,
                    n_head=12,
                    n_embd=768,
                    dropout=0,
                    bias=False,
                )
            ).to(device)
        finally:
            torch.set_default_dtype(old)
        _torch_load(model, init)
        model.train(not args.forward_only)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        result["devices"] = [device]
        result["torch_threads"] = torch.get_num_threads()
        assert (
            sum(p.numel() for p in model.parameters()) == result["parameters"]
        )
        if gpu:
            torch.mps.synchronize()
    result["setup_seconds_including_compile"] = time.perf_counter() - started
    for step, batch in data:
        tok, nxt = batch["tok"].values, batch["next"].values
        if args.backend.startswith("jax"):
            tokens, targets = jax.device_put(tok), jax.device_put(nxt)
            jax.block_until_ready((tokens, targets))
            begin = time.perf_counter()
            if args.forward_only:
                loss = function(params, tokens, targets)
            else:
                params, loss = function(params, tokens, targets)
            jax.block_until_ready((params, loss))
            elapsed = time.perf_counter() - begin
        else:
            tokens = torch.as_tensor(tok, device=device)
            targets = torch.as_tensor(nxt, device=device)
            if gpu:
                torch.mps.synchronize()
            begin = time.perf_counter()
            with torch.set_grad_enabled(not args.forward_only):
                if not args.forward_only:
                    optimizer.zero_grad(set_to_none=True)
                _, loss = model(tokens, targets)
                if not args.forward_only:
                    loss.backward()
                    optimizer.step()
            if gpu:
                torch.mps.synchronize()
            elapsed = time.perf_counter() - begin
            loss = loss.detach().cpu().item()
        if args.forward_only:
            want = n.jax_loss(reference, tok, nxt)
        else:
            want, gradients = value_grad(reference, tok, nxt)
            reference = {
                name: w - lr * gradients[name] for name, w in reference.items()
            }
        jax.block_until_ready((reference, want))
        errors = [
            n.assert_close("loss", loss, want, rtol=tolerance, atol=tolerance)
        ]
        if not args.forward_only:
            actual = (
                params.items()
                if args.backend.startswith("jax")
                else _torch_arrays(model)
            )
            for name, got in actual:
                errors.append(
                    n.assert_close(
                        name,
                        got,
                        reference[name],
                        rtol=tolerance,
                        atol=tolerance,
                    )
                )
        row = {
            "step": step,
            "seconds": elapsed,
            "loss": float(want),
            "check": "ok",
            "max_scaled_error": max(errors),
        }
        result["rows"].append(row)
        print(args.backend, row, flush=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    result["warm_seconds"] = (
        statistics.mean(r["seconds"] for r in result["rows"][1:])
        if args.steps > 1
        else result["rows"][0]["seconds"]
    )
    result["five_step_seconds"] = sum(r["seconds"] for r in result["rows"])
    result["five_steps_with_setup_compile_seconds"] = (
        result["five_step_seconds"] + result["setup_seconds_including_compile"]
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
