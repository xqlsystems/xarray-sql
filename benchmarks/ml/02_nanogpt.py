# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "xarray-sql[ml,duckdb]",
#   "jax",
# ]
#
# [tool.uv.sources]
# xarray-sql = { path = "../../", editable = true }
# ///
"""Demo 2: a GPT in SQL, trained by gradient descent.

nanoGPT (https://github.com/karpathy/nanoGPT) is Andrej Karpathy's small,
readable implementation of GPT-2, the language model. This is its model,
``model.py``, as one SQL query: the cross-entropy loss of a batch of
characters from Shakespeare, computed by a transformer whose parameters
are tables. ddx (https://github.com/xqlsystems/ddx), which differentiates
SQL queries, takes the loss's gradient with respect to every parameter,
so a training step is SQL too.

=================  ============================================
Model piece        In SQL
=================  ============================================
Embeddings         a join of the batch with ``wte`` and ``wpe``
Layer norm         per-position ``AVG``s, joined back, times a gain
Attention          per head and causal, as in ``01_attention.py``
MLP                two einsums around GELU
Output layer       an einsum with ``wte`` again (tied, as in GPT-2)
Loss               log-sum-exp per position, minus the target's logit
Gradients          ddx: ``grad(loss, w)``, one statement per table
Optimizer step     ``w - lr * grad(loss, w)``, the new table; or AdamW,
                   with Adam's moments as columns beside ``w`` and
                   the learning rate schedule as a table
=================  ============================================

As in nanoGPT, the model and the loss are separate. The model is the
view ``logits``; the loss is SQL that reads it, as nanoGPT's
``F.cross_entropy`` reads the model's logits.

Each training step updates every parameter table with its own
statement; the statements share one backward pass. By default the
optimizer is SGD at a fixed learning rate, the simplest step to read.
``--optimizer adamw`` trains as nanoGPT does instead: it stores the
gradients as tables, then updates each parameter table with AdamW. The same model in JAX
(Google's library for differentiable array programs), from the same
initial parameters and the same batches, is the reference: its loss at
every step, and its parameters after every step, must match the SQL's.

This is Part A of the second target demo of einfold
(https://github.com/xqlsystems/einfold), a query optimizer that rewrites
SQL plans into array algorithms; the demo is described in einfold's
``docs/demos.md``. Part B builds on it: it writes the tricks behind a
record of the NanoGPT speedrun (a public race to train GPT-2 fastest)
as a small diff to this SQL.

Differences from nanoGPT's ``model.py`` and ``train.py``:

* **Size.** nanoGPT's Shakespeare model has 6 layers, 6 heads, 384
  channels and 256 characters of context, 10.65M parameters. This one has
  1 layer, 2 heads, 16 channels and 16 characters, 4,416 parameters (the
  constants below).
* **GELU** is its tanh approximation, as GPT-2 computed it. nanoGPT's
  ``nn.GELU()`` uses the exact form, through the error function ``erf``,
  which DataFusion doesn't have. The two differ by less than 5e-4.
* **The optimizer**, by default, is SGD at a fixed learning rate.
  ``--optimizer adamw`` matches nanoGPT's: AdamW with its Shakespeare
  settings, weight decay on all but the layer norms, gradients clipped to
  a norm of 1, and the learning rate warming up, then decaying along a
  cosine. The schedule is stretched to ``--steps``, where nanoGPT's spans
  5,000. The JAX reference for this step matches ``torch.optim.AdamW``
  after ``clip_grad_norm_`` exactly.
* **No dropout**, nanoGPT's default, so that a run is deterministic and
  can be checked. Its Shakespeare configuration trains with 0.2.
* **Q, K and V are three tables.** nanoGPT's ``c_attn`` holds the same
  parameters as one matrix, split after the product.

Like nanoGPT's default (``bias = False`` in ``train.py``), the model has
no biases, in linear layers or layer norms.

Run ``uv run benchmarks/ml/02_nanogpt.py --help`` for the options.
"""

from __future__ import annotations

import pathlib
import urllib.request

import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr
from _bench import Report, assert_close, cli, hosts

import xarray_sql as xql

jax.config.update("jax_enable_x64", True)

# A tiny character-level GPT. The comments name each constant in nanoGPT's
# GPTConfig (https://github.com/karpathy/nanoGPT/blob/master/model.py) and
# its value in nanoGPT's Shakespeare model
# (https://github.com/karpathy/nanoGPT/blob/master/config/train_shakespeare_char.py).
LAYERS = 1  # n_layer = 6: transformer blocks
EMBED = 16  # n_embd = 384; e: channels of the residual stream
HEADS = 2  # n_head = 6; h: attention heads
HEAD = EMBED // HEADS  # c: channels per head
HIDDEN = 4 * EMBED  # f: channels of the MLP, 4 × n_embd as in nanoGPT
CONTEXT = 16  # block_size = 256; pos: characters per sequence
LEARNING_RATE = 0.5  # for SGD; nanoGPT's AdamW uses 1e-3, below

# AdamW (``--optimizer adamw``), with the settings of nanoGPT's Shakespeare
# model: train.py's defaults, and config/train_shakespeare_char.py's.
MAX_LR = 1e-3  # learning_rate
MIN_LR = 1e-4  # min_lr
WARMUP = 100 / 5000  # warmup_iters, as a fraction of max_iters
BETA1 = 0.9  # beta1
BETA2 = 0.99  # beta2
EPS = 1e-8  # torch.optim.AdamW's default
WEIGHT_DECAY = 0.1  # weight_decay, on every table but the layer norms' gains
GRAD_CLIP = 1.0  # grad_clip: the largest norm of all gradients together

# Each parameter table's dimensions; its values are the column w. In
# nanoGPT, wq, wk and wv are attn.c_attn; wo is attn.c_proj; w_fc and
# w_proj are mlp.c_fc and mlp.c_proj; ln_1, ln_2 and ln_f are the layer
# norms' weights.
PARAMS = {
    "wte": ("tok", "e"),
    "wpe": ("pos", "e"),
    "ln_1": ("layer", "e"),
    "wq": ("layer", "e", "h", "c"),
    "wk": ("layer", "e", "h", "c"),
    "wv": ("layer", "e", "h", "c"),
    "wo": ("layer", "h", "c", "e"),
    "ln_2": ("layer", "e"),
    "w_fc": ("layer", "e", "f"),
    "w_proj": ("layer", "f", "e"),
    "ln_f": ("e",),
}

# --------------------------------------------------------------------------
# The model, in SQL. Activations are tables of (seq, pos, e, val): one row
# per sequence in the batch, position, and channel.
# --------------------------------------------------------------------------


def layer_norm(x: str, out: str, gain: str, layer: int | None = None) -> str:
    """Normalize each position of ``x`` to zero mean and unit variance,
    then scale each channel by the table ``gain`` (of ``layer``, if set)."""
    of_layer = "" if layer is None else f" AND g.layer = {layer}"
    return f"""
{out}_mean AS (
  SELECT seq, pos, AVG(val) AS mean FROM {x} GROUP BY seq, pos),
{out}_var AS (
  SELECT x.seq, x.pos, AVG(power(x.val - m.mean, 2)) AS var
  FROM {x} x JOIN {out}_mean m ON x.seq = m.seq AND x.pos = m.pos
  GROUP BY x.seq, x.pos),
{out} AS (
  SELECT x.seq, x.pos, x.e, (x.val - m.mean) / sqrt(v.var + 1e-5) * g.w AS val
  FROM {x} x
  JOIN {out}_mean m ON x.seq = m.seq AND x.pos = m.pos
  JOIN {out}_var v ON x.seq = v.seq AND x.pos = v.pos
  JOIN {gain} g ON x.e = g.e{of_layer}),"""


# GELU's tanh approximation: x/2 · (1 + tanh(√(2/π) · (x + 0.044715 x³))).
GELU_SCALE = np.sqrt(2 / np.pi)


def block(layer: int, x: str, out: str) -> str:
    """One transformer block: ``out = x + attention(ln(x))``, then
    ``out = out + mlp(ln(out))``."""
    l = f"l{layer}"
    projections = "".join(
        f"""
{l}_{name} AS (
  SELECT x.seq, x.pos, p.h, p.c, SUM(x.val * p.w) AS val
  FROM {l}_ln1 x JOIN w{name} p ON x.e = p.e
  WHERE p.layer = {layer}
  GROUP BY x.seq, x.pos, p.h, p.c),"""
        for name in ("q", "k", "v")
    )
    return f"""{layer_norm(x, f"{l}_ln1", "ln_1", layer)}{projections}
{l}_scores AS (
  SELECT q.seq, q.h, q.pos AS i, k.pos AS j,
         SUM(q.val * k.val) / {np.sqrt(HEAD)} AS s
  FROM {l}_q q JOIN {l}_k k ON q.seq = k.seq AND q.h = k.h AND q.c = k.c
  WHERE q.pos >= k.pos
  GROUP BY q.seq, q.h, q.pos, k.pos),
{l}_weights AS (
  -- The max only keeps exp() from overflowing: it cancels out of p / z.
  SELECT seq, h, i, j,
         exp(s - MAX(s) OVER (PARTITION BY seq, h, i)) AS p
  FROM {l}_scores),
{l}_norm AS (
  SELECT seq, h, i, SUM(p) AS z FROM {l}_weights GROUP BY seq, h, i),
{l}_heads AS (
  SELECT a.seq, a.i AS pos, a.h, v.c, SUM(a.p / n.z * v.val) AS val
  FROM {l}_weights a
  JOIN {l}_v v ON a.seq = v.seq AND a.h = v.h AND a.j = v.pos
  JOIN {l}_norm n ON a.seq = n.seq AND a.h = n.h AND a.i = n.i
  GROUP BY a.seq, a.i, a.h, v.c),
{l}_attn AS (
  SELECT a.seq, a.pos, p.e, SUM(a.val * p.w) AS val
  FROM {l}_heads a JOIN wo p ON a.h = p.h AND a.c = p.c
  WHERE p.layer = {layer}
  GROUP BY a.seq, a.pos, p.e),
{l}_mid AS (
  SELECT x.seq, x.pos, x.e, x.val + a.val AS val
  FROM {x} x JOIN {l}_attn a ON x.seq = a.seq AND x.pos = a.pos AND x.e = a.e),
{layer_norm(f"{l}_mid", f"{l}_ln2", "ln_2", layer)}
{l}_fc AS (
  SELECT x.seq, x.pos, p.f, SUM(x.val * p.w) AS val
  FROM {l}_ln2 x JOIN w_fc p ON x.e = p.e
  WHERE p.layer = {layer}
  GROUP BY x.seq, x.pos, p.f),
{l}_hidden AS (
  SELECT seq, pos, f,
         0.5 * val * (1 + tanh({GELU_SCALE} * (val + 0.044715 * val * val * val)))
           AS val
  FROM {l}_fc),
{l}_mlp AS (
  SELECT x.seq, x.pos, p.e, SUM(x.val * p.w) AS val
  FROM {l}_hidden x JOIN w_proj p ON x.f = p.f
  WHERE p.layer = {layer}
  GROUP BY x.seq, x.pos, p.e),
{out} AS (
  SELECT x.seq, x.pos, x.e, x.val + m.val AS val
  FROM {l}_mid x JOIN {l}_mlp m ON x.seq = m.seq AND x.pos = m.pos AND x.e = m.e),"""


def model() -> str:
    """The model, as the view ``logits``: for each position of each
    sequence in ``batch``, a score for every character that could come
    next. A DataFusion view reads the tables that existed when it was
    created (DuckDB's looks them up by name), so it is recreated whenever
    the batch or the parameters are."""
    blocks = "".join(
        block(layer, f"x{layer}", f"x{layer + 1}") for layer in range(LAYERS)
    )
    return f"""
CREATE OR REPLACE VIEW logits AS
WITH x0 AS (
  SELECT b.seq, b.pos, te.e, te.w + pe.w AS val
  FROM batch b
  JOIN wte te ON b.tok = te.tok
  JOIN wpe pe ON b.pos = pe.pos AND te.e = pe.e),
{blocks}
{layer_norm(f"x{LAYERS}", "final", "ln_f").rstrip(",")}
SELECT x.seq, x.pos, te.tok, SUM(x.val * te.w) AS val
FROM final x JOIN wte te ON x.e = te.e
GROUP BY x.seq, x.pos, te.tok
"""


# The loss, separate from the model, as in nanoGPT: the mean cross-entropy
# of the model's logits against each next character of the batch.
# Subtracting each row's max only keeps exp() from overflowing: it cancels
# out of the log-sum-exp.
#
# The loss reads the model's view once, into the CTE ``predicted``, and
# uses that twice: a host expands a view at every read, but DuckDB
# computes a CTE that is read twice only once.
LOSS = """
predicted AS (
  SELECT seq, pos, tok, val FROM logits),
shifted AS (
  SELECT seq, pos, tok, val,
         MAX(val) OVER (PARTITION BY seq, pos) AS m
  FROM predicted),
log_sum_exp AS (
  SELECT seq, pos, MAX(m) + ln(SUM(exp(val - m))) AS lse
  FROM shifted GROUP BY seq, pos),
loss AS (
  SELECT AVG(s.lse - l.val) AS l
  FROM log_sum_exp s
  JOIN batch b ON s.seq = b.seq AND s.pos = b.pos
  JOIN predicted l ON l.seq = b.seq AND l.pos = b.pos AND l.tok = b.next)"""


def step_statements(lr: float) -> list[str]:
    """One SGD step: each parameter table's next version, as a statement."""
    return [
        f"WITH {LOSS} SELECT {', '.join(dims)}, "
        f"w - {lr} * grad(loss, w) AS w FROM {table} "
        f"ORDER BY {', '.join(dims)}"
        for table, dims in PARAMS.items()
    ]


def gradient_statements() -> list[str]:
    """The gradient of the loss with respect to each parameter table, as
    the table ``{name}_grad``'s rows. The statements share one backward
    pass."""
    return [
        f"WITH {LOSS} SELECT {', '.join(dims)}, grad(loss, w) AS g "
        f"FROM {table} ORDER BY {', '.join(dims)}"
        for table, dims in PARAMS.items()
    ]


def adamw_statement(table: str, step: int) -> str:
    """One AdamW step on a parameter table with columns ``w``, ``m`` and
    ``v`` (the weight and Adam's two moments), from its gradient in
    ``{table}_grad`` and the learning rate in ``schedule``. This is
    ``torch.optim.AdamW``, after nanoGPT clips the gradients."""
    dims = PARAMS[table]
    keys = ", ".join(f"p.{d}" for d in dims)
    same_row = " AND ".join(f"p.{d} = g.{d}" for d in dims)
    decay = 0.0 if table.startswith("ln_") else WEIGHT_DECAY
    every_grad = "\n    UNION ALL ".join(
        f"SELECT g FROM {t}_grad" for t in PARAMS
    )
    return f"""
WITH grad_norm AS (
  SELECT sqrt(SUM(g * g)) AS norm FROM (
    {every_grad})),
clip AS (
  SELECT LEAST(1.0, {GRAD_CLIP} / (norm + 1e-6)) AS c FROM grad_norm),
moments AS (
  SELECT {keys}, p.w,
         {BETA1} * p.m + (1 - {BETA1}) * g.g * clip.c AS m,
         {BETA2} * p.v + (1 - {BETA2}) * power(g.g * clip.c, 2) AS v
  FROM {table} p
  JOIN {table}_grad g ON {same_row}
  CROSS JOIN clip)
SELECT {keys},
       p.w * (1 - s.lr * {decay})
         - s.lr * (p.m / (1 - power({BETA1}, s.step + 1)))
           / (sqrt(p.v / (1 - power({BETA2}, s.step + 1))) + {EPS}) AS w,
       p.m,
       p.v
FROM moments p JOIN schedule s ON s.step = {step}
ORDER BY {keys}
"""


def loss_statement() -> str:
    return f"WITH {LOSS} SELECT l FROM loss"


# --------------------------------------------------------------------------
# The same model in JAX: the reference.
# --------------------------------------------------------------------------


def jax_layer_norm(x, gain):
    mean = x.mean(-1, keepdims=True)
    var = ((x - mean) ** 2).mean(-1, keepdims=True)
    return (x - mean) / jnp.sqrt(var + 1e-5) * gain


def jax_loss(p, tok, nxt):
    x = p["wte"][tok] + p["wpe"][None]
    causal = jnp.tril(jnp.ones((CONTEXT, CONTEXT), dtype=bool))
    for layer in range(LAYERS):
        h = jax_layer_norm(x, p["ln_1"][layer])
        q, k, v = (
            jnp.einsum("bte,ehc->bhtc", h, p[w][layer])
            for w in ("wq", "wk", "wv")
        )
        s = jnp.einsum("bhic,bhjc->bhij", q, k) / np.sqrt(HEAD)
        a = jax.nn.softmax(jnp.where(causal, s, -jnp.inf), axis=-1)
        heads = jnp.einsum("bhij,bhjc->bhic", a, v)
        x = x + jnp.einsum("bhtc,hce->bte", heads, p["wo"][layer])
        h = jax_layer_norm(x, p["ln_2"][layer])
        hidden = jax.nn.gelu(h @ p["w_fc"][layer], approximate=True)
        x = x + hidden @ p["w_proj"][layer]
    logits = jax_layer_norm(x, p["ln_f"]) @ p["wte"].T
    target = jnp.take_along_axis(logits, nxt[..., None], -1)[..., 0]
    return jnp.mean(jax.nn.logsumexp(logits, -1) - target)


def jax_adamw(params, moments, grads, step, lr):
    """The reference AdamW step: new parameters and moments."""
    norm = np.sqrt(sum(float(jnp.sum(g**2)) for g in grads.values()))
    clip = min(1.0, GRAD_CLIP / (norm + 1e-6))
    new_params, new_moments = {}, {}
    for name, w in params.items():
        g = grads[name] * clip
        m, v = moments[name]
        m = BETA1 * m + (1 - BETA1) * g
        v = BETA2 * v + (1 - BETA2) * g**2
        decay = 0.0 if name.startswith("ln_") else WEIGHT_DECAY
        m_hat = m / (1 - BETA1 ** (step + 1))
        v_hat = v / (1 - BETA2 ** (step + 1))
        new_params[name] = w * (1 - lr * decay) - lr * m_hat / (
            jnp.sqrt(v_hat) + EPS
        )
        new_moments[name] = (m, v)
    return new_params, new_moments


# --------------------------------------------------------------------------
# Data and parameters, as xarray Datasets.
# --------------------------------------------------------------------------

SHAKESPEARE = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/"
    "data/tinyshakespeare/input.txt"
)


def shakespeare() -> str:
    """Tiny Shakespeare, cached in ~/.cache; a sonnet offline."""
    cache = (
        pathlib.Path.home() / ".cache" / "xarray-sql" / "tinyshakespeare.txt"
    )
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        try:
            with urllib.request.urlopen(SHAKESPEARE, timeout=10) as r:
                cache.write_bytes(r.read())
        except OSError:
            return SONNET_18 * 4
    return cache.read_text()


def batches(text: str, size: int, rng):
    """Endless batches of (seq, pos) characters and the characters after
    them, as one Dataset: the table ``batch(seq, pos, tok, next)``."""
    vocab = sorted(set(text))
    ids = np.array([vocab.index(c) for c in text])
    while True:
        starts = rng.integers(0, len(ids) - CONTEXT - 1, size)
        tok = np.stack([ids[s : s + CONTEXT] for s in starts])
        nxt = np.stack([ids[s + 1 : s + CONTEXT + 1] for s in starts])
        yield xr.Dataset(
            {"tok": (("seq", "pos"), tok), "next": (("seq", "pos"), nxt)},
            coords={"seq": np.arange(size), "pos": np.arange(CONTEXT)},
        )


def init_params(vocab: int, rng) -> dict[str, np.ndarray]:
    """nanoGPT's initialization, from GPT-2: weights normal with standard
    deviation 0.02, or 0.02/√(2 × layers) for the two projections that add
    to the residual stream; layer norm gains 1."""
    sizes = {
        "tok": vocab,
        "pos": CONTEXT,
        "layer": LAYERS,
        "e": EMBED,
        "h": HEADS,
        "c": HEAD,
        "f": HIDDEN,
    }
    params = {}
    for name, dims in PARAMS.items():
        shape = [sizes[d] for d in dims]
        if name.startswith("ln_"):
            params[name] = np.ones(shape)
        elif name in ("wo", "w_proj"):
            params[name] = rng.normal(0, 0.02 / np.sqrt(2 * LAYERS), shape)
        else:
            params[name] = rng.normal(0, 0.02, shape)
    return params


def as_table(name: str, w: np.ndarray, m=None, v=None) -> xr.Dataset:
    """A parameter as a table: its weights ``w``, and Adam's moments ``m``
    and ``v`` when they are given."""
    dims = PARAMS[name]
    columns = {"w": w, "m": m, "v": v}
    return xr.Dataset(
        {k: (dims, x) for k, x in columns.items() if x is not None},
        coords={d: np.arange(n) for d, n in zip(dims, w.shape)},
    )


def as_array(name: str, rows, column: str = "w") -> np.ndarray:
    return xql.to_dataset(rows, dims=list(PARAMS[name]))[column].values


def schedule(steps: int) -> xr.Dataset:
    """nanoGPT's learning rate at each step, as the table
    ``schedule(step, lr)``: a linear warmup, then a cosine decay to the
    minimum, with train.py's ``max_iters`` and ``lr_decay_iters`` both
    ``steps``."""
    warmup = round(WARMUP * steps)
    lr = []
    for it in range(steps):
        if it < warmup:
            lr.append(MAX_LR * (it + 1) / (warmup + 1))
        else:
            decay = (it - warmup) / (steps - warmup)
            coeff = 0.5 * (1 + np.cos(np.pi * decay))
            lr.append(MIN_LR + coeff * (MAX_LR - MIN_LR))
    return xr.Dataset({"lr": ("step", lr)}, coords={"step": np.arange(steps)})


def adamw_step(host, step: int) -> list:
    """One AdamW step on every parameter table: the gradients, in one
    backward pass, registered as tables; then each table's update."""
    gradients = host.query_all(gradient_statements())
    for name, rows in zip(PARAMS, gradients):
        host.load(f"{name}_grad", xql.to_dataset(rows, dims=list(PARAMS[name])))
    return [host.query(adamw_statement(name, step)) for name in PARAMS]


def main():
    parser = cli(__doc__.splitlines()[0], sizes=[4], size_name="batch")
    parser.add_argument("--steps", type=int, default=5, help="training steps")
    parser.add_argument(
        "--optimizer",
        choices=["sgd", "adamw"],
        default="sgd",
        help="SGD at a fixed learning rate (default), or nanoGPT's AdamW",
    )
    args = parser.parse_args()
    adamw = args.optimizer == "adamw"
    lr = schedule(args.steps)
    report = Report("nanogpt", "batch", args.reps)
    text = shakespeare()
    vocab = len(set(text))
    lines = len(model().splitlines()) + len(loss_statement().splitlines())
    count = sum(
        v.size for v in init_params(vocab, np.random.default_rng(0)).values()
    )
    print(
        f"  {LAYERS}-layer GPT, vocab {vocab}, {count:,} parameters: "
        f"the loss is {lines} lines of SQL"
    )

    for size in args.sizes:
        init = init_params(vocab, np.random.default_rng(0))
        data = list(
            zip(
                range(args.steps), batches(text, size, np.random.default_rng(1))
            )
        )

        for host in hosts(args.hosts):
            reference = dict(init)
            moments = {
                n: (np.zeros_like(w), np.zeros_like(w)) for n, w in init.items()
            }
            for name, values in init.items():
                state = moments[name] if adamw else ()
                host.load(name, as_table(name, values, *state))
            host.load("schedule", lr)

            for step, batch in data:
                host.load("batch", batch)
                host.execute(model())
                tok, nxt = batch["tok"].values, batch["next"].values

                rows = report.time(
                    f"loss, step {step}",
                    host,
                    size,
                    host.query,
                    loss_statement(),
                )
                want = jax_loss(reference, tok, nxt)
                report.check(
                    assert_close("loss", rows["l"][0].as_py(), want, rtol=1e-9)
                )
                report.note("loss", round(float(want), 6))
                report.note("operators", host.operators(loss_statement()))

                train = f"train ({args.optimizer}), step {step}"
                if not host.differentiates:
                    report.skip(train, host, size, "no ddx backend")
                    continue
                if adamw:
                    updated = report.attempt(
                        train, host, size, adamw_step, host, step
                    )
                else:
                    updated = report.attempt(
                        train,
                        host,
                        size,
                        host.query_all,
                        step_statements(LEARNING_RATE),
                    )
                if updated is None:
                    # Every later step starts from this one's weights.
                    break
                grads = jax.grad(jax_loss)(reference, tok, nxt)
                if adamw:
                    reference, moments = jax_adamw(
                        reference, moments, grads, step, lr["lr"].values[step]
                    )
                else:
                    reference = {
                        name: w - LEARNING_RATE * grads[name]
                        for name, w in reference.items()
                    }

                errors = []
                for name, rows in zip(PARAMS, updated):
                    expected = {"w": reference[name]}
                    if adamw:
                        expected["m"], expected["v"] = moments[name]
                    for column, values in expected.items():
                        errors.append(
                            assert_close(
                                f"{name}.{column}, step {step}",
                                as_array(name, rows, column),
                                values,
                            )
                        )
                    host.load(
                        name, xql.to_dataset(rows, dims=list(PARAMS[name]))
                    )
                report.check(max(errors))

    report.finish(args.csv)


SONNET_18 = """Shall I compare thee to a summer's day?
Thou art more lovely and more temperate:
Rough winds do shake the darling buds of May,
And summer's lease hath all too short a date;
Sometime too hot the eye of heaven shines,
And often is his gold complexion dimm'd;
And every fair from fair sometime declines,
By chance or nature's changing course untrimm'd;
But thy eternal summer shall not fade,
Nor lose possession of that fair thou ow'st;
Nor shall death brag thou wander'st in his shade,
When in eternal lines to time thou grow'st:
So long as men can breathe or eyes can see,
So long lives this, and this gives life to thee.
"""


if __name__ == "__main__":
    main()
