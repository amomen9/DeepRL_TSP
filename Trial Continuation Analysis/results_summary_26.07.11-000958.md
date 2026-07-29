# Returns summary - mean/std across all repetitions and eval points (n_repetitions=6)

| Algorithm | Setting | Mean (all) | Std (all) | Mean (last 10%) | Std (last 10%) | N (all) | N (last) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PPO | PPO, γ:0.99, A-α:3e-4, A-NN:[128,128], Full-Ep:✓, $λ_{GAE}$:0.96 | -3,049.018 | 323.987 | -2,781.948 | 287.396 | 12,000 | 1,200 |
| PPO | PPO, γ:0.99, A-α:3e-4, A-NN:[128,128], Full-Ep:✗, $λ_{GAE}$:0.96 | -3,446.627 | 262.005 | -3,335.298 | 272.467 | 12,000 | 1,200 |
