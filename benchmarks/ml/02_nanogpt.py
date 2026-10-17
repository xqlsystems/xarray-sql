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
"""Demo 2: a GPT in SQL, trained by gradient descent through ddx.

nanoGPT is Andrej Karpathy's small, readable GPT-2. This is its model as
one SQL query: the cross-entropy loss of a batch of characters from
Shakespeare, computed by a transformer whose parameters are tables.

=================  ============================================
Model piece        In SQL
=================  ============================================
Embeddings         a join of the batch with ``wte`` and ``wpe``
Layer norm         per-position ``AVG``s, joined back
Attention          Demo 1's query, per head, causal
MLP                two einsums around ReLU², ``power(greatest(x, 0), 2)``
Output layer       an einsum with ``wte`` again (tied, as in GPT-2)
Loss               log-sum-exp per position, minus the target's logit
Gradients          ddx: ``grad(loss, w)``, one statement per table
Optimizer step     ``w - lr * grad(loss, w)``, the new table
=================  ============================================

Each training step updates every parameter table with its own
statement; the statements share one backward pass. The same model in JAX,
from the same initial parameters and the same batches, is the reference:
its loss at every step and its gradients must match the SQL's.

This is Part A of einfold's Demo 2 (``docs/demos.md``). Part B, the four
tricks of the NanoGPT speedrun's sparsity record as a small diff to this
SQL, builds on it. Simplifications from nanoGPT: no dropout, no biases,
and layer norms without a learned gain.

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

# A tiny character-level GPT.
LAYERS = 1  # transformer blocks
EMBED = 16  # e: channels of the residual stream
HEADS = 2  # h: attention heads
HEAD = EMBED // HEADS  # c: channels per head
HIDDEN = 4 * EMBED  # f: channels of the MLP
CONTEXT = 16  # pos: characters per sequence
LEARNING_RATE = 0.5

# Each parameter table's dimensions; its values are the column w.
PARAMS = {
    "wte": ("tok", "e"),
    "wpe": ("pos", "e"),
    "wq": ("layer", "e", "h", "c"),
    "wk": ("layer", "e", "h", "c"),
    "wv": ("layer", "e", "h", "c"),
    "wo": ("layer", "h", "c", "e"),
    "w_fc": ("layer", "e", "f"),
    "w_proj": ("layer", "f", "e"),
}

# --------------------------------------------------------------------------
# The model, in SQL. Activations are tables of (seq, pos, e, val): one row
# per sequence in the batch, position, and channel.
# --------------------------------------------------------------------------


def layer_norm(x: str, out: str) -> str:
    """Normalize each position of ``x`` to zero mean and unit variance."""
    return f"""
{out}_mean AS (
  SELECT seq, pos, AVG(val) AS mean FROM {x} GROUP BY seq, pos),
{out}_var AS (
  SELECT x.seq, x.pos, AVG(power(x.val - m.mean, 2)) AS var
  FROM {x} x JOIN {out}_mean m ON x.seq = m.seq AND x.pos = m.pos
  GROUP BY x.seq, x.pos),
{out} AS (
  SELECT x.seq, x.pos, x.e, (x.val - m.mean) / sqrt(v.var + 1e-5) AS val
  FROM {x} x
  JOIN {out}_mean m ON x.seq = m.seq AND x.pos = m.pos
  JOIN {out}_var v ON x.seq = v.seq AND x.pos = v.pos),"""


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
    return f"""{layer_norm(x, f"{l}_ln1")}{projections}
{l}_scores AS (
  SELECT q.seq, q.h, q.pos AS i, k.pos AS j,
         SUM(q.val * k.val) / {np.sqrt(HEAD)} AS s
  FROM {l}_q q JOIN {l}_k k ON q.seq = k.seq AND q.h = k.h AND q.c = k.c
  WHERE q.pos >= k.pos
  GROUP BY q.seq, q.h, q.pos, k.pos),
{l}_weights AS (
  SELECT seq, h, i, j,
         exp(s - ddx_stop_gradient(MAX(s) OVER (PARTITION BY seq, h, i))) AS p
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
{layer_norm(f"{l}_mid", f"{l}_ln2")}
{l}_hidden AS (
  SELECT x.seq, x.pos, p.f, power(greatest(SUM(x.val * p.w), 0.0), 2) AS val
  FROM {l}_ln2 x JOIN w_fc p ON x.e = p.e
  WHERE p.layer = {layer}
  GROUP BY x.seq, x.pos, p.f),
{l}_mlp AS (
  SELECT x.seq, x.pos, p.e, SUM(x.val * p.w) AS val
  FROM {l}_hidden x JOIN w_proj p ON x.f = p.f
  WHERE p.layer = {layer}
  GROUP BY x.seq, x.pos, p.e),
{out} AS (
  SELECT x.seq, x.pos, x.e, x.val + m.val AS val
  FROM {l}_mid x JOIN {l}_mlp m ON x.seq = m.seq AND x.pos = m.pos AND x.e = m.e),"""


def model() -> str:
    """The CTEs of the whole model, ending in ``loss``: the mean
    cross-entropy of predicting each next character of the batch."""
    blocks = "".join(
        block(layer, f"x{layer}", f"x{layer + 1}") for layer in range(LAYERS)
    )
    return f"""
x0 AS (
  SELECT b.seq, b.pos, te.e, te.w + pe.w AS val
  FROM batch b
  JOIN wte te ON b.tok = te.tok
  JOIN wpe pe ON b.pos = pe.pos AND te.e = pe.e),
{blocks}
{layer_norm(f"x{LAYERS}", "final")}
logits AS (
  SELECT x.seq, x.pos, te.tok, SUM(x.val * te.w) AS val
  FROM final x JOIN wte te ON x.e = te.e
  GROUP BY x.seq, x.pos, te.tok),
shifted AS (
  SELECT seq, pos, tok, val,
         ddx_stop_gradient(MAX(val) OVER (PARTITION BY seq, pos)) AS m
  FROM logits),
log_sum_exp AS (
  SELECT seq, pos, MAX(m) + ln(SUM(exp(val - m))) AS lse
  FROM shifted GROUP BY seq, pos),
loss AS (
  SELECT AVG(s.lse - l.val) AS l
  FROM log_sum_exp s
  JOIN batch b ON s.seq = b.seq AND s.pos = b.pos
  JOIN logits l ON l.seq = b.seq AND l.pos = b.pos AND l.tok = b.next)"""


def step_statements(lr: float) -> list[str]:
    """One SGD step: each parameter table's next version, as a statement."""
    return [
        f"WITH {model()} SELECT {', '.join(dims)}, "
        f"w - {lr} * grad(loss, w) AS w FROM {table} "
        f"ORDER BY {', '.join(dims)}"
        for table, dims in PARAMS.items()
    ]


def loss_statement() -> str:
    return f"WITH {model()} SELECT l FROM loss"


# --------------------------------------------------------------------------
# The same model in JAX: the reference.
# --------------------------------------------------------------------------


def jax_layer_norm(x):
    mean = x.mean(-1, keepdims=True)
    var = ((x - mean) ** 2).mean(-1, keepdims=True)
    return (x - mean) / jnp.sqrt(var + 1e-5)


def jax_loss(p, tok, nxt):
    x = p["wte"][tok] + p["wpe"][None]
    causal = jnp.tril(jnp.ones((CONTEXT, CONTEXT), dtype=bool))
    for layer in range(LAYERS):
        h = jax_layer_norm(x)
        q, k, v = (
            jnp.einsum("bte,ehc->bhtc", h, p[w][layer])
            for w in ("wq", "wk", "wv")
        )
        s = jnp.einsum("bhic,bhjc->bhij", q, k) / np.sqrt(HEAD)
        a = jax.nn.softmax(jnp.where(causal, s, -jnp.inf), axis=-1)
        heads = jnp.einsum("bhij,bhjc->bhic", a, v)
        x = x + jnp.einsum("bhtc,hce->bte", heads, p["wo"][layer])
        h = jax_layer_norm(x)
        hidden = jnp.maximum(h @ p["w_fc"][layer], 0.0) ** 2
        x = x + hidden @ p["w_proj"][layer]
    logits = jax_layer_norm(x) @ p["wte"].T
    target = jnp.take_along_axis(logits, nxt[..., None], -1)[..., 0]
    return jnp.mean(jax.nn.logsumexp(logits, -1) - target)


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
    """GPT-2's initialization: normal, standard deviation 0.02."""
    sizes = {
        "tok": vocab,
        "pos": CONTEXT,
        "layer": LAYERS,
        "e": EMBED,
        "h": HEADS,
        "c": HEAD,
        "f": HIDDEN,
    }
    return {
        name: rng.normal(0, 0.02, [sizes[d] for d in dims])
        for name, dims in PARAMS.items()
    }


def as_table(name: str, values: np.ndarray) -> xr.Dataset:
    dims = PARAMS[name]
    return xr.Dataset(
        {"w": (dims, values)},
        coords={d: np.arange(n) for d, n in zip(dims, values.shape)},
    )


def as_array(name: str, rows) -> np.ndarray:
    return xql.to_dataset(rows, dims=list(PARAMS[name]))["w"].values


def main():
    parser = cli(__doc__.splitlines()[0], sizes=[4], size_name="batch")
    parser.add_argument("--steps", type=int, default=5, help="SGD steps")
    args = parser.parse_args()
    report = Report("nanogpt", "batch", args.reps)
    text = shakespeare()
    vocab = len(set(text))
    lines = len(loss_statement().splitlines())
    print(
        f"  {LAYERS}-layer GPT, vocab {vocab}: the loss is {lines} lines of SQL"
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
            for name, values in init.items():
                host.load(name, as_table(name, values))

            for step, batch in data:
                host.load("batch", batch)
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

                if not host.differentiates:
                    report.skip(
                        f"train, step {step}", host, size, "no ddx backend"
                    )
                    continue
                updated = report.time(
                    f"train, step {step}",
                    host,
                    size,
                    host.query_all,
                    step_statements(LEARNING_RATE),
                )
                grads = jax.grad(jax_loss)(reference, tok, nxt)
                errors = []
                for name, rows in zip(PARAMS, updated):
                    reference[name] = (
                        reference[name] - LEARNING_RATE * grads[name]
                    )
                    trained = as_array(name, rows)
                    errors.append(
                        assert_close(
                            f"{name}, step {step}", trained, reference[name]
                        )
                    )
                    host.load(name, as_table(name, trained))
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
