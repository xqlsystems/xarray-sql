You're taking over the large experiments for "sparse Samudra", a Samudra-style ocean emulator written in SQL, where land cells have no rows at any layer. The goal is accurate forecasts, scored with Samudra's own `samudra.metrics`. It does not need to match Samudra's weights. The bar is Samudra 2 at the same resolution. Another agent is training a standard 2° Samudra 2 run as that baseline.

Read first:
- The modeling notes and early results: `benchmarks/ml/SPARSE_SAMUDRA_NOTES.md` on branch `sparse-samudra-notes` of https://github.com/xqlsystems/xarray-sql
- The code: branch `samudra-demo` (draft PR https://github.com/xqlsystems/xarray-sql/pull/300).
  - `benchmarks/ml/04_samudra_sparse.py` is the model. It includes the SQL, a PyTorch "mirror" that trains it, and a dense ablation.
  - `benchmarks/ml/_samudra_metrics.py` is the scorer. Pull first; it's being committed now.
  - `uv run benchmarks/ml/04_samudra_sparse.py --help` lists the options. Key ones: `--data om4`, `--depths all`, `--widths`, `--unroll`, `--by-change`, `--dense`, `--device cuda`, `--train START END`, `--test START END`, `--rollout`, `--sql-steps`, `--save DIR`, `--json FILE`. A `--resolution 2|1` flag is being added.
- Samudra's Torch guide: https://m2lines.github.io/Samudra/docs/torch/. Also read Samudra's AGENTS.md, especially its ocean-state invariants and gotchas.

Setup on Torch:
- Work in `/scratch/$USER`. Use the cluster's OM4 copies; don't copy data from OSN.
  - The script streams from `https://nyu1.osn.mghpcc.org/m2lines-pubs/Samudra/v2026-09/om4_{twodeg,onedeg}/`. Add a flag to read a local path, and say which dataset version you used. The cluster path in the guide is `om4_onedeg_v3`, which may differ from v2026-09.
- The script is a uv script, and its header pins PyTorch's CPU index.
  - For GPUs, point the header's torch index at a CUDA build that matches the nodes, for example `https://download.pytorch.org/whl/cu126`.
  - xarray-sql builds from source (Rust, with maturin). If the container can't build it, run the PyTorch parts there, and run the SQL check where a build works.
- Follow the guide's Slurm settings for account, partition, and CPU and memory per GPU. One GPU is enough for everything below except possibly step 5.

Experiments, in order. Score each with `samudra.metrics` over Samudra's inference period (rollout from 2014-10-10 to 2022-12-24, scored 2015 to 2022). Always train the dense ablation alongside, the same way.
1. **Reproduce at 2°** with the defaults, and check you get roughly the numbers in the notes.
2. **2°, all depths:** `--depths all --widths 32 64`, training on Samudra's split, 1975-01-03 to 2013-09-30, with `--unroll 2` and then 4. Try `--by-change` on and off. All depths are needed to score ocean heat content.
3. **2°, with velocities** (`uo`, `vo`) and wider: `--widths 64 128 256`, `--unroll 4`, 40 epochs. Compare with the 2° Samudra 2 baseline once it exists.
4. **1°:** repeat steps 2 and 3 with `--resolution 1`.
   - The 1° grid has 180 rows. Check that the land rows are cropped and that the U-Net halves cleanly at each level.
   - Baseline: score the public Samudra 2 checkpoint `M2LInES/Samudra2` `onedeg/ema_ckpt.pt` on Hugging Face with the same metrics.
5. **1° at Samudra 2's widths** (280, 380, 480, 520; 4 levels), using its loss and rollout length if you can match them.
6. **No retraining:** load the 1° Samudra 2 checkpoint into Samudra's own model, mask land after every layer (or use the sparse operators), and score it. This shows how much the trained model relies on land.

For every run:
- **Keep the SQL check:** `--sql-steps 1 --hosts datafusion`. At widths of 128 and above, one SQL forecast is slow, roughly an hour at Samudra 2's widths. In that case run `--sql-steps 0` on the big run, and run the check once on a small copy with the same code.
- **Check the physics on each rollout:** heat budget, salinity drift, area-mean `zos` near 0, temperature at or above freezing, and stratification.
- **Remember Samudra's gotchas:**
  - `areacello` is in m² despite its label. Use it only as relative weights.
  - `samudra.metrics` uses c_p = 3850, while Samudra's physics constants use 3992.

Reporting:
- **Commits:** push code changes to `samudra-demo`, in small commits. No force-pushes, and nothing to `main`. Keep PR #300 a draft.
- **Results:** post on PR #300, one comment per experiment. Include:
  - the command;
  - the git commit;
  - GPU type and count, wall time, and package versions;
  - the dataset path and version, and the train and test dates;
  - the `samudra.metrics` table for sparse, dense and the baseline;
  - the physics checks.

  Report bad results too.
- **Large files:** keep weights and rollouts on Torch scratch, and give their paths in the comment. Don't commit them.
- **Notes:** update `SPARSE_SAMUDRA_NOTES.md` on `sparse-samudra-notes` as the results come in.

Rules:
- No credentials, hostnames beyond the public docs, or internal paths outside `/scratch` in anything you push or post.
- The repo's AGENTS.md applies: introduce every reference, no issue numbers in code comments, and keep the SQL legible.
- If you find a bug in xarray-sql, ddx or einfold, file an issue with a verified repro.
