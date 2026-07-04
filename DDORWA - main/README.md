# Stochastic TSP with Random Traffic Delay

This project extends the Travelling Salesman Problem (TSP) by adding random directed traffic delay between selected nodes.  A problem instance is defined by three matrices:

```text
distance_matrix      base deterministic travel time
max_delay_matrix     maximum possible delay on every directed edge
delay_mask           1 only where delay is active, 0 otherwise
```

Built-in instances may store `uncertain_routes`, a 1-indexed directed edge list used to build the same delay mask too.  The delay is uniform by default:

```text
delay(i,j) ~ Uniform(0, max_delay_matrix[i,j])
```

The compared methods are:

```text
VI              exact expected-model value iteration; skipped safely for large n
Classic DP      deterministic Held-Karp solver on the base matrix; skipped safely for large n
RL              REINFORCE / AC / A2C / PPO trained by stochastic interaction
PointerNet      Pointer Network with attention + LSTM decoder, trained with A2C
Heuristic       risk-aware non-learning policy using base matrix + delay locations
```

RL training is step-budgeted and episode-updated: `--timesteps` controls total environment steps, and each policy update happens after a complete tour.

## Files

```text
env.py                   stochastic TSP environment
instances.py             built-ins, JSON loader, random matrix generator
value_iteration.py       expected-model Bellman value iteration
classic_DP.py            deterministic Held-Karp DP
RL_policy_based.py       REINFORCE, AC, A2C, PPO agents
RL_pointer_network.py    Pointer Network agent (attention + LSTM decoder, A2C update)
RL_trainer.py            generic training loop
heuristic.py             risk-aware heuristic
helper.py                multi-seed RL experiment + curve plotting
EXP_ALL.py               fair final comparison
EXP_RL.py                RL learning / HPO curves
```

## Instances

List built-ins:

```bat
python EXP_ALL.py --list-instances
```

Default built-ins:

```text
1  Example_4_nodes
2  Example_5_nodes
3  Capacity_6_delay_between_3_nodes
4  Capacity_10_delay_between_4_nodes
```

If no instance is specified, both experiment scripts use the first four built-ins.

## Random capacity tests

Generate one random n-node instance:

```bat
python EXP_ALL.py --random 20 --timesteps 5000 --runs 30
```

Specify directed delayed edges manually.  These are not symmetric: `1,2` does not imply `2,1`.

```bat
python EXP_ALL.py --random 10 --directions 1,2 3,4 4,7 --timesteps 5000 --runs 30
```

If `--directions` is omitted, the generator selects about 10--15% of the full matrix size as directed delay-active paths.  The base matrix is random symmetric; the max-delay matrix is full and scaled to the base durations; the delay mask controls where delay actually happens.  Random experiments print all three matrices.

## `EXP_ALL.py`: fair comparison

Default:

```bat
python EXP_ALL.py
```

Logic:

```text
--random N given      use one generated N-node instance; print matrices
--instances given    use selected built-in / JSON instances
nothing given        use built-in ids 1..4
both given           random instance wins
```

Common commands:

```bat
python EXP_ALL.py --quick
python EXP_ALL.py --instances 3 --method A2C --timesteps 10000 --runs 100 --seed 42
python EXP_ALL.py --instances 1 3 4 --method AC --timesteps 5000 --runs 50
python EXP_ALL.py --random 25 --method A2C --timesteps 20000 --runs 50 --exact-time-min 5
```

Pointer Network:

```bat
python EXP_ALL.py --pointer-net --random 20 --timesteps 20000 --runs 30
python EXP_ALL.py --pointer-net --method A2C --instances 3 --timesteps 50000 --runs 50
```

Pointer Network flags:

```text
--pointer-net              include the Pointer Network agent in training and evaluation
--pointer-embed-dim N      embedding dimension (default 64)
--pointer-glimpses N       number of glimpse steps (default 1)
```

Safety for exact methods:

```text
--exact-time-min 5        terminate each VI/DP run after 5 minutes
--exact-memory-gb 8       optional explicit memory budget for one VI/DP run
--exact-memory-frac 0.70  default: allow 70% of currently available RAM
```

There is no hand-crafted node-size cutoff.  Before VI or Classic DP starts, the code estimates the exponential state storage.  If the estimate exceeds the memory budget, the method is reported as skipped.  If the method runs longer than the time limit, it is terminated and reported as timed out.  RL and heuristic continue running either way.

## `EXP_RL.py`: RL learning and HPO

Default:

```bat
python EXP_RL.py
```

This runs the default single RL config on the default selected instances.  Use `--quick` first for a smoke test:

```bat
python EXP_RL.py --quick
```

Single config, no sweep:

```bat
python EXP_RL.py --instances 3 --method A2C --timesteps 10000 --seeds 0 1 2 3 4
```

Actor learning-rate sweep:

```bat
python EXP_RL.py --instances 3 --method A2C --sweep actor_lr --actor-lrs 0.0001 0.001 0.01 --timesteps 10000 --seeds 0 1 2 3 4
```

Critic learning-rate sweep:

```bat
python EXP_RL.py --instances 3 --method AC --sweep critic_lr --critic-lrs 0.0003 0.001 0.003 --timesteps 10000
```

Actor-network sweep:

```bat
python EXP_RL.py --instances 3 --method REINFORCE --sweep actor_hidden --actor-hidden-values 32 32,32 64,64 --timesteps 10000
```

A2C n-step sweep:

```bat
python EXP_RL.py --instances 3 --method A2C --sweep n_step --n-step-values 1 3 5 10 --timesteps 10000
```

Available sweeps:

```text
actor_lr, critic_lr, actor_hidden, critic_hidden, entropy, gamma, n_step
```

If `--sweep` is not provided, the script runs one config using the base hyperparameters.  If `--sweep` is provided, the sweep takes priority.

## Custom JSON instances

```json
{
  "instances": [
    {
      "id": "5",
      "name": "My_case",
      "distance_matrix": [[0, 10], [10, 0]],
      "max_delay_matrix": [[0, 4], [4, 0]],
      "uncertain_routes": [[1, 2]],
      "delay_distribution": "uniform"
    }
  ]
}
```

Run:

```bat
python EXP_ALL.py --instance-file instances.json --instances 5 --timesteps 10000 --runs 100
python EXP_RL.py --instance-file instances.json --instances 5 --method A2C --sweep actor_lr
```

## PPO note

`PPO` is available as an additional policy-gradient method.  Unlike REINFORCE, AC, and A2C, PPO intentionally uses its own internal GAE, clipped objective, mini-batches, and several training epochs.  No extra PPO-specific controller is exposed; use the existing `--n-envs` and `--minibatch-size` controls when running PPO.

Example:

```bash
python EXP_RL.py --instances 3 --method PPO --timesteps 10000 --n-envs 8 --minibatch-size 64
```
