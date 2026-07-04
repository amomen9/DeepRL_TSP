# DDORWA — Data-Driven Optimization of Routing With Aleatoric/epistemic uncertainty

Reinforcement-learning study on a **stochastic Travelling Salesman Problem**
(A2C / PPO / SAC vs. exact dynamic programming and a Pointer-Network baseline).
A TSP instance is defined by three matrices held on the environment:

- `duration_matrix` (D) — deterministic base edge durations,
- `uncertainty_inclusion_matrix` (I) — which edges carry uncertainty,
- `potential_uncertainty_matrix` (U) — per-edge upper bound of the noise.

## Disk reuse and instance matching

Results (`data sheets/<ALGO>.xlsx`) and network checkpoints
(`Checkpoints/TSP/<ALGO>/`) are reused across runs only when they were produced
for the **same instance**. On top of the existing matching rules (project
config + per-sheet hyperparameters for Excel; `n_actions` + architecture for
checkpoints), every disk load also keys on the three instance-defining matrices
above:

- the full D / I / U matrices are stored as text (provenance) and compared via a
  **SHA1 hash signature**;
- **Excel results** are always matched on the matrix hash;
- **checkpoints** are matched on the matrix hash when
  `checkpoints.match_training_matrices` is `True` (default), so a saved
  actor/critic is reused only if it was trained on the same instance — enforced
  even under loose / Net2Net-resize matching.

---

## Bundled submodule: `TSP-DRL_Bello` — fork modifications

The `TSP-DRL_Bello/` submodule is the Pointer-Network baseline ("Bello").

**Original (upstream) repository:** `ci-ke/TSP-DRL_Bello` —
<https://github.com/ci-ke/TSP-DRL_Bello> (itself a typed/refactored fork of
Rintaro Kobayashi's original implementation `Rintarooo/TSP_DRL_PtrNet` —
<https://github.com/Rintarooo/TSP_DRL_PtrNet>, a PyTorch implementation of
Bello et al. 2016, *Neural Combinatorial Optimization with Reinforcement
Learning*, <https://arxiv.org/abs/1611.09940>).

This project's fork (`amomen9/TSP-DRL_Bello`) diverges from upstream at commit
`147ad84` ("pass mypy check"). Every change made since that fork commit is
listed below (the same report is kept in the submodule's own README).

### 1. Checkpoint save location (commit `b8fe271`)

- **`config.py`** — `model_dir` now defaults to
  `<project_root>/Checkpoints/TSP-DRL_Bello/` (via the new `DEFAULT_MODEL_DIR`)
  instead of `./Pt/`, so trained actors land in the parent project's central
  checkpoint tree.
- **`train.py`** — comment updated for the new default model directory.
- **`Csv/1113_12_12_train20_param.csv`** — regenerated parameter log.

### 2. Learning from a duration matrix

Lets the same Pointer Network train/infer on a fixed, possibly **asymmetric**
duration matrix (a DDORWA instance) instead of only random 2-D Euclidean
coordinates, while leaving the original coordinate behaviour intact and keeping
coordinates available for plotting. All changes are **additive and
backward-compatible**: with the default `input_dim = 2` every original code path
behaves exactly as before.

- **`config.py`** — new hyperparameter `input_dim` (default `2`) with a
  `-id/--input_dim` CLI flag and a backward-compat guard. `2` = original
  `(x, y)` coordinate encoder; `2*city_t` = duration-matrix encoder.
- **`actor.py` (`PtrNet1`)** / **`critic.py` (`PtrNet2`)** — the input embedding
  is now `nn.Linear(cfg.input_dim, cfg.embed)`; with `input_dim = 2` the
  architecture is identical to the original.
- **`env.py` (`Env_tsp`)** — additive methods (coordinate methods untouched):
  `set_duration_matrix(D)`; `matrix_node_features` (per city: outgoing **row**
  concat incoming **column** → `(city_t, 2*city_t)`); `stack_matrix_nodes`
  (batch of one fixed instance); `stack_l_matrix` (**directed** tour cost
  `Σ D[t_i, t_{i+1}] + D[t_last, t_0]`).

**Symmetric-equivalence guarantee:** for a symmetric input matrix the directed
objective and mirrored node features coincide with the undirected/coordinate
formulation, so the modified Bello optimises the same problem the original did
(verified numerically against the original Euclidean `stack_l_fast`).

### 3. Encoder-reference projection precompute (performance)

A pure-speed change to the actor/critic forward passes; **results are
unchanged**. The Pointer Network attends over the encoder outputs `ref = enc_h`,
which are **fixed for the whole decode**. The original code nonetheless
recomputed their `Conv1d` projections *inside every step*:

- **`actor.py` (`PtrNet1`)** — `glimpse` recomputed `W_ref(ref)` and `pointer`
  recomputed `W_ref2(ref)` at **each of the `city_t` decoder steps**.
- **`critic.py` (`PtrNet2`)** — `glimpse` recomputed `W_ref(ref)` at **each of
  the `n_process` glimpse steps**.

Because `ref` never changes during decoding, those projections are now computed
**once per forward**, before the loop, and reused; `glimpse`/`pointer` take the
precomputed `(batch, hidden, city_t)` tensor instead of `ref`. This removes the
redundant `Conv1d` work — which is `O(city_t²)` per forward while everything else
is `O(city_t)` — so the speedup **grows with instance size**.

- **Measured** (CPU, 1 thread, batch 512 — the conditions the baseline runs
  under): **≈2.0× faster at 20 cities** (~50% less wall-time per training step),
  **≈2.9× at 50 cities**.
- **Correctness:** the forward is **bit-identical** to the original — same tours,
  same log-likelihoods, same critic values; gradients match to float round-off
  (a reused shared graph node vs. per-step re-created nodes differ only in
  summation order). The benchmark curve is unaffected.
- **Regression guard:** `test_ptr_precompute.py` pins this — it asserts the
  optimised forward equals the original recompute-every-step logic (forward
  bit-identical, gradients within tolerance) and exits non-zero on divergence
  (`python test_ptr_precompute.py` from the submodule).

This mirrors what the POMO Attention Model (the `TSP-DRL-Test` submodule) already
does by construction — compute the decoder keys/values once per instance and
reuse them across all decode steps — bringing Bello's forward in line with it.

### Parent-side integration

Setting `global_config['baseline_model'] = 'Bello'` runs the baseline through the
main experiment runner ([Library/Library_bello_baseline.py](Library/Library_bello_baseline.py)):

- **Trains on the same fixed `duration_matrix`** under the main-repo standards
  (`eval_interval`, `n_timesteps`, `n_repetitions`, `max_episode_length`,
  `base_seed`) with Bello's immutable hyperparameters; `n_repetitions`
  independent retrainings are aggregated into one mean **benchmark curve** drawn
  against the RL learning curves. Bello's `steps` budget is mapped onto the
  shared env-step x-grid so the curve overlays the RL curves.
- **Disk reuse** mirrors the other algorithms: results are saved to / loaded
  from `data sheets/BELLO.xlsx` and per-rep actors to `Checkpoints/TSP/BELLO/`,
  matched on the project config, the instance-matrix hash and the per-sheet
  hyperparameters. For Bello `use_existing_disk_data` is **always honoured as
  True** — a matching workbook short-circuits retraining regardless of the global
  flag.
- **Plotting** reconstructs 2-D coordinates from the best tour found, via the
  route-preserving symmetrisation + classical-MDS embedding in
  [Library/Library_bello_plot.py](Library/Library_bello_plot.py)
  (`<plots>/BELLO_route_reconstructed.png`). This coordinate reconstruction is a
  reusable utility and works for any algorithm's found route.

---

## Bundled submodule: `TSP-DRL-Test` — POMO Attention Model baseline

The `TSP-DRL-Test/` submodule is the second, stronger policy-network contender:
the **Attention Model** (Kool, van Hoof & Welling, ICLR 2019) trained with
**POMO** — *Policy Optimization with Multiple Optima* (Kwon et al., NeurIPS
2020). It targets the same policy-learning task as Bello but with a modern,
critic-free recipe, and is designed to beat it at a much smaller budget.

**Repository:** `amomen9/TSP-DRL-Test` —
<https://github.com/amomen9/TSP-DRL-Test>. Unlike the Bello submodule this is not
a fork of a single upstream repo but an independent implementation of AM + POMO;
the full method write-up lives in the submodule's own README.

### How it differs from the original (Bello) approach

A tour is still built city by city from a stochastic policy factorised by the
chain rule, exactly as in Bello et al. — but every component around it is
upgraded:

| Component | Bello et al. 2017 (`TSP-DRL_Bello`) | `TSP-DRL-Test` (AM + POMO) |
|---|---|---|
| Encoder | LSTM over the city *sequence* (imposes an ordering on a set) | Multi-head self-attention stack (permutation-invariant, no recurrence) |
| Decoder | LSTM + 1 attention glimpse | Context attention (graph + first + last node), multi-head glimpse, clipped single-head pointer |
| REINFORCE baseline | Learned critic network (extra params, extra MSE loss, biased early) | **POMO shared baseline** — mean cost of `pomo_size` rollouts of the *same* instance forced to start from distinct cities; unbiased, **no critic** |
| Samples per update | `batch` independent tours | `batch × pomo_size` structurally diverse multi-start tours |
| Per-step encoder work | LSTM re-encodes every decode step | Embeddings + decoder keys/values computed **once**, reused every step (the same optimisation Bello change #3 above now applies) |
| Inference | sampling / active search | greedy multi-start from every city, optional ×8 symmetry augmentation |

Retained from Bello et al. because they still help: tanh logit clipping
`C·tanh(u)` (`clip_logits`) and the softmax-temperature knob (`softmax_T`).

### Matrix mode and parent-side integration

- **Duration-matrix API.** `env.py` exposes the *same* matrix interface as the
  Bello fork (`set_duration_matrix` / `matrix_node_features` /
  `stack_matrix_nodes` / `stack_l_matrix`: node features = outgoing **row** concat
  incoming **column**, **directed** tour cost on the matrix), so the AM/POMO
  model trains on the identical fixed instance the RL agents and Bello see.
- **Runner integration.** `Experiment.py` enables the method with
  `included_algorithms["TSP_TEST"] = True`; **all** of its hyperparameters live
  in the `tsp_test_config` section (overridable on the CLI via
  `--set tsp_test.KEY=VALUE`). It can also be selected as the benchmark curve via
  `baseline_model = "TSP_TEST"`.
  [Library/Library_tsp_test_experiment.py](Library/Library_tsp_test_experiment.py)
  trains it on the same fixed `duration_matrix`, maps its gradient-step budget
  onto the shared env-step x-grid, and draws its curve next to the A2C/PPO/SAC
  curves and the Bello benchmark.
- **Disk reuse** mirrors the other algorithms: results to
  `data sheets/TSP_TEST.xlsx` and checkpoints to `Checkpoints/TSP/TSP_TEST/`,
  matched on the project config, the instance-matrix hash and the per-sheet
  hyperparameters.
- **No module-name clashes.** The host imports this repo as a *package* through
  `importlib` (see its `__init__.py`), so its `config` / `env` / `train` module
  names never collide with the Bello submodule's identically named flat modules.
