"""Train nanoGPT's Shakespeare character model with SQL and ddx gradients.

nanoGPT (https://github.com/karpathy/nanoGPT) supplies the training recipe
and prepared character data. The model, loss, gradient clipping and AdamW
updates execute SQL; EinFold (https://github.com/xqlsystems/einfold) lowers
the plans to compact CPU arrays or MLX (https://github.com/ml-explore/mlx)
GPU arrays. Python samples batches and dropout masks and writes checkpoints.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import resource
import subprocess
import time
from pathlib import Path

import ddxdb
import numpy as np
import xarray as xr
from _hosts import DenseCPU, DenseGPU
from _shakespeare_torch import _TorchReference

import xarray_sql as xql
from xarray_sql.math import register_erf
from xarray_sql.training import TrainingSession

_SPEC = importlib.util.spec_from_file_location(
    "_shakespeare_sql_model", Path(__file__).with_name("02_nanogpt.py")
)
_MODEL = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODEL)

# The maximum cancels out of cross entropy. Spell its reduction as a join
# to avoid repeatedly traversing a shared model graph during window lowering.
# Holding this shared shift constant removes only a cancelling derivative.
_MODEL.LOSS = """
predicted AS (SELECT seq,pos,tok,val FROM logits),
maximum AS (SELECT seq,pos,ddx_stop_gradient(MAX(val)) AS m FROM predicted GROUP BY seq,pos),
log_sum_exp AS (
  SELECT p.seq,p.pos,MAX(m.m)+ln(SUM(exp(p.val-m.m))) AS lse
  FROM predicted p JOIN maximum m ON p.seq=m.seq AND p.pos=m.pos
  GROUP BY p.seq,p.pos),
loss AS (
  SELECT AVG(s.lse-l.val) AS l FROM log_sum_exp s
  JOIN batch b ON s.seq=b.seq AND s.pos=b.pos
  JOIN predicted l ON l.seq=b.seq AND l.pos=b.pos AND l.tok=b.next)
"""


def _parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--backend", choices=["dense_cpu", "dense_gpu"], default="dense_gpu"
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=Path.home() / "nanoGPT/data/shakespeare_char",
    )
    p.add_argument("--out-dir", type=Path, default=Path("out-shakespeare-sql"))
    p.add_argument(
        "--resume",
        type=Path,
        help="resume an NPZ checkpoint including moments and RNGs",
    )
    p.add_argument(
        "--steps",
        type=int,
        help="cap this run's updates without shortening the LR schedule",
    )
    p.add_argument(
        "--verify-nanogpt",
        type=Path,
        help="check every gradient, update and moment against this nanoGPT checkout",
    )
    for name, default in {
        "batch_size": 12,
        "block_size": 256,
        "n_layer": 6,
        "n_head": 6,
        "n_embd": 384,
        "gradient_accumulation_steps": 1,
        "max_iters": 2000,
        "lr_decay_iters": 2000,
        "warmup_iters": 100,
        "eval_interval": 200,
        "eval_iters": 20,
        "log_interval": 10,
        "seed": 1337,
    }.items():
        p.add_argument(
            "--" + name,
            "--" + name.replace("_", "-"),
            type=int,
            default=default,
        )
    for name, default in {
        "dropout": 0.2,
        "learning_rate": 1e-3,
        "min_lr": 1e-4,
        "beta1": 0.9,
        "beta2": 0.99,
        "weight_decay": 0.1,
        "grad_clip": 1.0,
    }.items():
        p.add_argument(
            "--" + name,
            "--" + name.replace("_", "-"),
            type=float,
            default=default,
        )
    p.add_argument("--cache-gib", type=float, default=16)
    p.add_argument(
        "--resident",
        action="store_true",
        help="keep SQL training state as owned device arrays",
    )
    p.add_argument(
        "--compile-training",
        action="store_true",
        help="compile the complete SQL-derived device step",
    )
    p.add_argument(
        "--gpu-dropout",
        action="store_true",
        help="generate dropout masks inside the device training step",
    )
    p.add_argument(
        "--profile-memory",
        action="store_true",
        help="sample current RSS outside training timers at log intervals",
    )
    return p


def _learning_rate(it, args):
    if it < args.warmup_iters:
        return args.learning_rate * (it + 1) / (args.warmup_iters + 1)
    if it > args.lr_decay_iters:
        return args.min_lr
    ratio = (it - args.warmup_iters) / (args.lr_decay_iters - args.warmup_iters)
    return args.min_lr + 0.5 * (1 + math.cos(math.pi * ratio)) * (
        args.learning_rate - args.min_lr
    )


def _data(args):
    # Match prepare.py's uint16 files and sorted character vocabulary without
    # executing the pickle in meta.pkl. Check that the files encode input.txt.
    text = (args.data_dir / "input.txt").read_text()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    encoded = np.fromiter((stoi[c] for c in text), dtype=np.uint16)
    split = int(len(encoded) * 0.9)
    data = {}
    for name, expected in (
        ("train", encoded[:split]),
        ("val", encoded[split:]),
    ):
        values = np.memmap(
            args.data_dir / f"{name}.bin", dtype=np.uint16, mode="r"
        )
        if not np.array_equal(values, expected):
            raise ValueError(
                f"{name}.bin differs from nanoGPT's character preparation of input.txt"
            )
        if len(values) <= args.block_size:
            raise ValueError(f"{name} split is shorter than the context")
        data[name] = values
    return chars, data


def _batch(values, args, rng):
    starts = rng.integers(len(values) - args.block_size, size=args.batch_size)
    tok = np.stack([values[i : i + args.block_size] for i in starts]).astype(
        np.int64
    )
    nxt = np.stack(
        [values[i + 1 : i + args.block_size + 1] for i in starts]
    ).astype(np.int64)
    return xr.Dataset(
        {"tok": (("seq", "pos"), tok), "next": (("seq", "pos"), nxt)},
        coords={
            "seq": np.arange(args.batch_size),
            "pos": np.arange(args.block_size),
        },
    )


def _masks(args, rng):
    def mask(dims, shape):
        scale = (rng.random(shape) >= args.dropout).astype(np.float64) / (
            1 - args.dropout
        )
        return xr.Dataset(
            {"scale": (dims, scale)},
            coords={d: np.arange(n) for d, n in zip(dims, shape)},
        )

    shape = (args.batch_size, args.block_size, args.n_embd)
    result = {"embed_drop": mask(("seq", "pos", "e"), shape)}
    for layer in range(args.n_layer):
        result[f"l{layer}_adrop"] = mask(
            ("seq", "h", "i", "j"),
            (args.batch_size, args.n_head, args.block_size, args.block_size),
        )
        for suffix in ("rdrop", "mdrop"):
            result[f"l{layer}_{suffix}"] = mask(("seq", "pos", "e"), shape)
    return result


def _clip_sql(args):
    if args.grad_clip == 0:
        return "SELECT 1.0 AS c"
    every_grad = " UNION ALL ".join(
        f"SELECT g FROM {t}_grad" for t in _MODEL.PARAMS
    )
    return f"""WITH norm AS (SELECT sqrt(SUM(g*g)) AS n FROM ({every_grad})),
    coefficient AS (SELECT {args.grad_clip}/(n+1e-6) AS c FROM norm)
    SELECT CASE WHEN c < 1.0 THEN c ELSE 1.0 END AS c FROM coefficient"""


def _update_sql(name, args):
    dims = _MODEL.PARAMS[name]
    keys = ", ".join(f"p.{d}" for d in dims)
    same = lambda alias: " AND ".join(f"p.{d}={alias}.{d}" for d in dims)
    decay = 0.0 if name.startswith("ln_") else args.weight_decay
    return f"""WITH moments AS (
    SELECT {keys}, p.w,
      {args.beta1}*s.m + (1.0-{args.beta1})*g.g*c.c AS m,
      {args.beta2}*s.v + (1.0-{args.beta2})*power(g.g*c.c,2) AS v
    FROM {name} p JOIN {name}_grad g ON {same("g")}
    JOIN {name}_state s ON {same("s")} CROSS JOIN clip c)
    SELECT {keys},
      p.w*(1.0-t.lr*{decay}) - t.lr*(p.m/t.b1)/(sqrt(p.v/t.b2)+1e-8) AS w,
      p.m, p.v FROM moments p CROSS JOIN train_step t ORDER BY {keys}"""


def _save(
    path,
    params,
    moments,
    iteration,
    args,
    chars,
    data_rng,
    mask_rng,
    best,
    skip_eval=False,
):
    metadata = {
        "iteration": iteration,
        "config": vars(args)
        | {
            "bias": False,
            "dtype": "float32" if args.backend == "dense_gpu" else "float64",
        },
        "chars": chars,
        "data_rng": data_rng.bit_generator.state,
        "mask_rng": mask_rng.bit_generator.state,
        "best_val_loss": best,
        "skip_eval": skip_eval,
    }
    arrays = {
        f"{name}_{col}": value
        for name, w in params.items()
        for col, value in zip(("w", "m", "v"), (w, *moments[name]))
    }
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez(
            handle,
            metadata=np.array(json.dumps(metadata, default=str)),
            **arrays,
        )
    temporary.replace(path)


def main():
    parser = _parser()
    args = parser.parse_args()
    if args.gpu_dropout:
        args.resident = True
        if args.backend != "dense_gpu":
            parser.error("--gpu-dropout requires dense_gpu")
    if args.compile_training:
        args.resident = True
        if args.backend != "dense_gpu":
            parser.error("--compile-training requires dense_gpu")
    if (
        min(
            args.batch_size,
            args.block_size,
            args.n_layer,
            args.n_head,
            args.n_embd,
            args.eval_interval,
            args.eval_iters,
            args.log_interval,
        )
        < 1
        or args.n_embd % args.n_head
    ):
        parser.error(
            "model sizes and evaluation/logging intervals must be positive, with n_embd divisible by n_head"
        )
    if args.gradient_accumulation_steps != 1:
        parser.error(
            "this recipe currently supports gradient_accumulation_steps=1"
        )
    if (
        not 0 <= args.beta1 < 1
        or not 0 <= args.beta2 < 1
        or not all(
            math.isfinite(v) and v >= 0
            for v in (
                args.learning_rate,
                args.min_lr,
                args.weight_decay,
                args.grad_clip,
            )
        )
        or not math.isfinite(args.cache_gib)
        or args.cache_gib <= 0
    ):
        parser.error(
            "require betas in [0,1), finite nonnegative optimizer values and a positive cache size"
        )
    if (
        not 0 <= args.dropout < 1
        or not 0 <= args.warmup_iters < args.lr_decay_iters
        or args.max_iters < 0
        or (args.steps is not None and args.steps < 1)
    ):
        parser.error(
            "require 0 <= dropout < 1, 0 <= warmup_iters < lr_decay_iters and positive steps"
        )
    if args.backend == "dense_gpu" and os.environ.get("MLX_ENABLE_TF32") != "0":
        parser.error(
            "GPU execution requires MLX_ENABLE_TF32=0 before Python starts"
        )
    if "erf" not in ddxdb.supported_functions():
        parser.error(
            "exact GELU needs the local ddx math build: python accelerators/ddx_math/build.py"
        )
    _MODEL.LAYERS, _MODEL.EMBED, _MODEL.HEADS = (
        args.n_layer,
        args.n_embd,
        args.n_head,
    )
    _MODEL.HEAD, _MODEL.HIDDEN, _MODEL.CONTEXT = (
        args.n_embd // args.n_head,
        4 * args.n_embd,
        args.block_size,
    )
    chars, data = _data(args)
    data_rng, mask_rng = (
        np.random.default_rng(args.seed),
        np.random.default_rng(args.seed + 1),
    )
    params = _MODEL.init_params(len(chars), np.random.default_rng(args.seed))
    moments = {
        name: (np.zeros_like(w), np.zeros_like(w)) for name, w in params.items()
    }
    iteration, best, skip_eval = 0, math.inf, False
    if args.resume:
        with np.load(args.resume, allow_pickle=False) as saved:
            meta = json.loads(str(saved["metadata"]))
            ignored = {
                "steps",
                "resume",
                "out_dir",
                "data_dir",
                "verify_nanogpt",
                "cache_gib",
                "max_iters",
                "profile_memory",
                "resident",
                "compile_training",
            }
            for key, value in vars(args).items():
                if (
                    key not in ignored
                    and meta["config"].get(key, False) != value
                ):
                    parser.error(f"checkpoint configuration differs for {key}")
            if meta["chars"] != chars:
                parser.error("checkpoint vocabulary differs from input.txt")
            for name, w in params.items():
                for col in ("w", "m", "v"):
                    if saved[f"{name}_{col}"].shape != w.shape:
                        parser.error(
                            f"checkpoint shape differs for {name}_{col}"
                        )
                params[name] = saved[f"{name}_w"].copy()
                moments[name] = tuple(
                    saved[f"{name}_{c}"].copy() for c in ("m", "v")
                )
            iteration, best, skip_eval = (
                meta["iteration"],
                meta["best_val_loss"],
                meta["skip_eval"],
            )
            data_rng.bit_generator.state = meta["data_rng"]
            mask_rng.bit_generator.state = meta["mask_rng"]
    if args.verify_nanogpt and args.resume:
        parser.error(
            "reference verification starts from scratch; resume preserves SQL optimizer state"
        )
    os.environ["XQL_DENSE_CACHE_BYTES"] = str(int(args.cache_gib * (1 << 30)))
    host = DenseGPU() if args.backend == "dense_gpu" else DenseCPU()
    register_erf(host.ctx)
    for name, w in params.items():
        host.load(name, _MODEL.as_table(name, w))
        host.load(
            name + "_state",
            _MODEL.as_table(name, w, *moments[name]).drop_vars("w"),
        )
    # Schema registration must not advance the checkpointed RNG streams.
    host.load("batch", _batch(data["train"], args, np.random.default_rng(0)))
    for name, value in _masks(args, np.random.default_rng(0)).items():
        host.load(name, value)
    host.execute(
        _MODEL.model(exact_gelu=True, dropout=True, grouped_softmax=True)
    )
    # Register the second view directly: both graphs persist across evaluation.
    eval_model = _MODEL.model(
        exact_gelu=True, view="eval_logits", grouped_softmax=True
    )
    host.executor.register_view("eval_logits", eval_model.split(" AS\n", 1)[1])
    resident = None
    resident_step = None
    if args.resident:
        resident = TrainingSession(
            host.ctx,
            precision=host.dense_precision,
            max_cached_bytes=int(args.cache_gib * (1 << 30)),
            inline_ctes=host.executor.inline_ctes,
        )
        for name, query in host.executor._views.items():
            resident.executor.register_view(name, query)
        for name in host._tables:
            resident.bind(name)
        if args.compile_training:
            resident.index_domains(
                "batch", {"tok": ("wte", "tok"), "next": ("wte", "tok")}
            )
            resident.bind("batch")
        if args.gpu_dropout:
            resident.dropout(
                [name for name in host._tables if name.endswith("drop")],
                probability=args.dropout,
                seed=args.seed + 1,
                step=iteration,
            )
        updates_sql = {
            name + "_next": _update_sql(name, args) for name in params
        }
        assignments = {}
        for name, dims in _MODEL.PARAMS.items():
            assignments[name] = (name + "_next", [*dims, "w"])
            assignments[name + "_state"] = (name + "_next", [*dims, "m", "v"])
        resident_step = resident.prepare_step(
            gradients=dict(
                zip(
                    (name + "_grad" for name in params),
                    _MODEL.gradient_statements(),
                )
            ),
            stages={"clip": _clip_sql(args)},
            updates=updates_sql,
            assignments=assignments,
            compile=args.compile_training,
        )

    def load(name, value):
        if resident is None:
            host.load(name, value)
        else:
            resident.bind(name, value)

    def download():
        if resident is None:
            return
        tables = resident.export(*params, *(name + "_state" for name in params))
        for name in params:
            params[name] = _MODEL.as_array(name, tables[name])
            state = xql.to_dataset(
                tables[name + "_state"], dims=list(_MODEL.PARAMS[name])
            )
            moments[name] = (state.m.values, state.v.values)

    train_loss = host.executor.prepare(_MODEL.loss_statement())
    eval_loss = host.executor.prepare(
        _MODEL.loss_statement().replace("FROM logits", "FROM eval_logits")
    )
    gradients = host.executor.prepare_all(_MODEL.gradient_statements())
    clip = host.executor.prepare(_clip_sql(args))
    updates = {
        name: host.executor.prepare(_update_sql(name, args)) for name in params
    }
    reference = (
        _TorchReference(args, params, len(chars))
        if args.verify_nanogpt
        else None
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(
        json.dumps(
            {
                "backend": args.backend,
                "parameters": sum(w.size for w in params.values()),
                "vocabulary": len(chars),
                "config": vars(args),
            },
            default=str,
        ),
        flush=True,
    )
    # train.py tests iter_num > max_iters after each update, so iteration
    # max_iters is included. --steps provides an exact short-run update cap.
    end = args.max_iters + 1
    if args.steps is not None:
        end = min(end, iteration + args.steps)
    with (args.out_dir / "metrics.jsonl").open(
        "a" if args.resume else "w"
    ) as log:
        while iteration < end:
            if iteration % args.eval_interval == 0 and not skip_eval:
                losses = {}
                for split in ("train", "val"):
                    measured = []
                    for _ in range(args.eval_iters):
                        batch = _batch(data[split], args, data_rng)
                        load("batch", batch)
                        value = float(
                            (
                                eval_loss()
                                if resident is None
                                else resident.query(
                                    _MODEL.loss_statement().replace(
                                        "FROM logits", "FROM eval_logits"
                                    )
                                )
                            )["l"][0].as_py()
                        )
                        if reference:
                            reference.evaluate(batch, value)
                        measured.append(value)
                    losses[split] = float(np.mean(measured))
                event = dict(kind="eval", iteration=iteration, **losses)
                print(json.dumps(event), flush=True)
                log.write(json.dumps(event) + "\n")
                log.flush()
                if losses["val"] < best:
                    best = losses["val"]
                    if iteration > 0:
                        download()
                        _save(
                            args.out_dir / "best.npz",
                            params,
                            moments,
                            iteration,
                            args,
                            chars,
                            data_rng,
                            mask_rng,
                            best,
                            skip_eval=True,
                        )
            skip_eval = False
            batch = _batch(data["train"], args, data_rng)
            masks = {} if args.gpu_dropout else _masks(args, mask_rng)
            load("batch", batch)
            for name, value in masks.items():
                load(name, value)
            lr = _learning_rate(iteration, args)
            load(
                "train_step",
                xr.Dataset(
                    {
                        "lr": xr.DataArray(lr),
                        "b1": xr.DataArray(1 - args.beta1 ** (iteration + 1)),
                        "b2": xr.DataArray(1 - args.beta2 ** (iteration + 1)),
                    }
                ),
            )
            started = time.perf_counter()
            if resident is None:
                loss = float(train_loss()["l"][0].as_py())
                grad_rows = gradients()
                grads = {
                    name: xql.to_dataset(rows, dims=list(_MODEL.PARAMS[name]))
                    for name, rows in zip(params, grad_rows)
                }
                for name, ds in grads.items():
                    host.load(name + "_grad", ds)
                host.load("clip", xql.to_dataset(clip(), dims=[]))
                # Compute every update before replacing any current weight.
                next_tables = {
                    name: xql.to_dataset(run(), dims=list(_MODEL.PARAMS[name]))
                    for name, run in updates.items()
                }
                for name, ds in next_tables.items():
                    params[name] = ds.w.values
                    moments[name] = (ds.m.values, ds.v.values)
                    host.load(name, ds[["w"]])
                    host.load(name + "_state", ds[["m", "v"]])
            else:
                result = resident_step()
                loss = result.loss_value()
            seconds = time.perf_counter() - started
            if reference:
                if resident is not None:
                    if args.gpu_dropout:
                        mask_tables = result.export(
                            *(
                                name
                                for name in host._tables
                                if name.endswith("drop")
                            )
                        )
                        masks = {
                            name: xql.to_dataset(
                                rows,
                                dims=[
                                    f for f in rows.column_names if f != "scale"
                                ],
                            )
                            for name, rows in mask_tables.items()
                        }
                    grad_tables = result.export(
                        *(name + "_grad" for name in params)
                    )
                    grads = {
                        name: xql.to_dataset(
                            grad_tables[name + "_grad"],
                            dims=list(_MODEL.PARAMS[name]),
                        )
                        for name in params
                    }
                    download()
                reference.step(batch, masks, lr, loss, grads, params, moments)
            event = {
                "kind": "train",
                "iteration": iteration,
                "loss": loss,
                "lr": lr,
                "seconds": seconds,
                "tokens_per_second": args.batch_size
                * args.block_size
                / seconds,
                "verified": reference is not None,
            }
            if resident is not None:
                event["device_step"] = result.measurements
            if args.profile_memory and (
                iteration % args.log_interval == 0 or iteration + 1 == end
            ):
                event["rss_bytes"] = (
                    int(
                        subprocess.check_output(
                            ["ps", "-p", str(os.getpid()), "-o", "rss="],
                            text=True,
                        ).strip()
                    )
                    * 1024
                )
                event["peak_rss_bytes"] = resource.getrusage(
                    resource.RUSAGE_SELF
                ).ru_maxrss
                if (
                    resident is not None
                    and resident.executor._runtime is not None
                ):
                    event["mlx_memory"] = resident.executor._runtime.memory()
            log.write(json.dumps(event) + "\n")
            log.flush()
            if iteration % args.log_interval == 0 or iteration + 1 == end:
                print(json.dumps(event), flush=True)
            iteration += 1
    download()
    _save(
        args.out_dir / "last.npz",
        params,
        moments,
        iteration,
        args,
        chars,
        data_rng,
        mask_rng,
        best,
    )
    print(
        json.dumps(
            {
                "checkpoint": str(args.out_dir / "last.npz"),
                "next_iteration": iteration,
                "measurements": host.measurements()
                if resident is None
                else resident.statistics(),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
