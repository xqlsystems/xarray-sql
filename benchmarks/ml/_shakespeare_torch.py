"""Optional numerical oracle using the original nanoGPT model and AdamW.

nanoGPT (https://github.com/karpathy/nanoGPT) runs the same batches,
initial weights and externally supplied Bernoulli dropout masks. Only
dropout sampling and the fused attention selection are adapted, allowing
each SQL gradient and optimizer state to be checked independently.
"""

import importlib.util
import sys

import numpy as np

try:
    import torch
except ImportError:
    torch = None


class _TorchReference:
    def __init__(self, args, params, vocab):
        if torch is None:
            raise ImportError(
                "--verify-nanogpt requires PyTorch in this Python environment"
            )
        spec = importlib.util.spec_from_file_location(
            "_shakespeare_native_model", args.verify_nanogpt / "model.py"
        )
        native = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = native
        spec.loader.exec_module(native)
        torch.set_num_threads(8)
        self.args = args
        self.dtype = (
            torch.float32 if args.backend == "dense_gpu" else torch.float64
        )
        self.tolerance = 3e-5 if self.dtype == torch.float32 else 1e-8
        self.model = native.GPT(
            native.GPTConfig(
                block_size=args.block_size,
                vocab_size=vocab,
                n_layer=args.n_layer,
                n_head=args.n_head,
                n_embd=args.n_embd,
                dropout=args.dropout,
                bias=False,
            )
        ).to(dtype=self.dtype)
        self.masks = {}
        owner = self

        class _Mask(torch.nn.Module):
            def __init__(self, name):
                super().__init__()
                self.name = name

            def forward(self, x):
                return x * owner.masks[self.name] if self.training else x

        self.model.transformer.drop = _Mask("embed_drop")
        for layer, block in enumerate(self.model.transformer.h):
            block.attn.flash = False
            block.attn.register_buffer(
                "bias",
                torch.tril(torch.ones(args.block_size, args.block_size)).view(
                    1, 1, args.block_size, args.block_size
                ),
            )
            block.attn.attn_dropout = _Mask(f"l{layer}_adrop")
            block.attn.resid_dropout = _Mask(f"l{layer}_rdrop")
            block.mlp.dropout = _Mask(f"l{layer}_mdrop")
        with torch.no_grad():
            for name in params:
                for layer, target, encode in self._parts(name):
                    source = (
                        params[name] if layer is None else params[name][layer]
                    )
                    target.copy_(
                        torch.as_tensor(encode(source), dtype=self.dtype)
                    )
        decay = [p for p in self.model.parameters() if p.dim() >= 2]
        gains = [p for p in self.model.parameters() if p.dim() < 2]
        self.optimizer = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": args.weight_decay},
                {"params": gains, "weight_decay": 0.0},
            ],
            lr=args.learning_rate,
            betas=(args.beta1, args.beta2),
            eps=1e-8,
            foreach=False,
            fused=False,
        )

    def _parts(self, name):
        if name in ("wte", "wpe", "ln_f"):
            yield None, self.model.transformer[name].weight, lambda x: x
            return
        for layer, block in enumerate(self.model.transformer.h):
            if name in ("wq", "wk", "wv"):
                start = ("wq", "wk", "wv").index(name) * self.args.n_embd
                target = block.attn.c_attn.weight[
                    start : start + self.args.n_embd
                ]
                yield (
                    layer,
                    target,
                    lambda x: x.reshape(self.args.n_embd, self.args.n_embd).T,
                )
            elif name == "wo":
                yield (
                    layer,
                    block.attn.c_proj.weight,
                    lambda x: x.reshape(self.args.n_embd, self.args.n_embd).T,
                )
            elif name in ("w_fc", "w_proj"):
                attr = "c_fc" if name == "w_fc" else "c_proj"
                yield layer, getattr(block.mlp, attr).weight, lambda x: x.T
            else:
                yield layer, getattr(block, name).weight, lambda x: x

    def _arrays(self, name, column=None):
        def value(parameter):
            if column is None:
                return parameter.detach().numpy()
            if column == "grad":
                return parameter.grad.detach().numpy()
            return self.optimizer.state[parameter][column].detach().numpy()

        if name in ("wte", "wpe", "ln_f"):
            return value(self.model.transformer[name].weight)
        result = []
        for block in self.model.transformer.h:
            if name in ("wq", "wk", "wv"):
                start = ("wq", "wk", "wv").index(name) * self.args.n_embd
                result.append(
                    value(block.attn.c_attn.weight)[
                        start : start + self.args.n_embd
                    ].T.reshape(
                        self.args.n_embd,
                        self.args.n_head,
                        self.args.n_embd // self.args.n_head,
                    )
                )
            elif name == "wo":
                result.append(
                    value(block.attn.c_proj.weight).T.reshape(
                        self.args.n_head,
                        self.args.n_embd // self.args.n_head,
                        self.args.n_embd,
                    )
                )
            elif name in ("w_fc", "w_proj"):
                result.append(
                    value(
                        getattr(
                            block.mlp, "c_fc" if name == "w_fc" else "c_proj"
                        ).weight
                    ).T
                )
            else:
                result.append(value(getattr(block, name).weight))
        return np.stack(result)

    def _close(self, actual, expected, label):
        np.testing.assert_allclose(
            actual,
            expected,
            atol=self.tolerance,
            rtol=self.tolerance,
            err_msg=label,
        )

    def evaluate(self, batch, loss):
        self.model.eval()
        with torch.no_grad():
            _, expected = self.model(
                torch.from_numpy(batch.tok.values),
                torch.from_numpy(batch.next.values),
            )
        self._close(loss, float(expected), "evaluation loss")

    def step(self, batch, masks, lr, loss, grads, params, moments):
        self.model.train()
        self.masks = {
            name: torch.as_tensor(ds.scale.values, dtype=self.dtype)
            for name, ds in masks.items()
        }
        self.optimizer.zero_grad(set_to_none=True)
        _, expected = self.model(
            torch.from_numpy(batch.tok.values),
            torch.from_numpy(batch.next.values),
        )
        self._close(
            loss, float(expected.detach()), "training loss with dropout"
        )
        expected.backward()
        for name in params:
            self._close(
                grads[name].g.values,
                self._arrays(name, "grad"),
                f"{name} gradient",
            )
        if self.args.grad_clip != 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.args.grad_clip
            )
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.optimizer.step()
        for name in params:
            self._close(params[name], self._arrays(name), f"{name} weight")
            for values, col in zip(moments[name], ("exp_avg", "exp_avg_sq")):
                self._close(values, self._arrays(name, col), f"{name} {col}")
