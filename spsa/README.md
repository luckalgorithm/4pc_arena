# SPSA tuner

Use `tune.py` to tune UCI spin options for a four-player-chess engine with
simultaneous perturbation stochastic approximation (SPSA).

### Engine and match

| Flag | What it does | Default |
|---|---|---:|
| `-h`, `--help` | Shows command help. | |
| `--engine PATH` | Sets the UCI engine executable. | required |
| `--engine-options PATH` | Loads base engine options from JSON. | none |
| `--engine-option NAME=VALUE` | Sets a base engine option; repeatable. | none |
| `--arbiter PATH` | Uses a dedicated rules engine. | none |
| `--arbiter-options PATH` | Loads arbiter options from JSON. | none |
| `--arbiter-option NAME=VALUE` | Sets an arbiter option; repeatable. | none |
| `--arbiter-hash N` | Sets arbiter hash in MB. | `16` |
| `--threads N` | Sets threads per tested engine process. | `1` |
| `--hash N` | Sets hash in MB per tested engine process. | `128` |
| `--workers N` | Sets the number of games run concurrently. | up to `4` |
| `--nodes N` | Sets nodes per move. | `10000`* |
| `--depth N` | Sets fixed search depth per move. | none |
| `--movetime N` | Sets fixed milliseconds per move. | none |
| `--tc N` | Sets base clock time in milliseconds. | none |
| `--inc N` | Sets clock increment in milliseconds; requires `--tc`. | `0` |
| `--margin N` | Sets extra milliseconds allowed before a time loss. | `50` |
| `--timeout N` | Sets maximum seconds for an engine response. | `30.0` |
| `--max-plies N`, `--maxmoves N` | Draws after N total plies. | `1000` |
| `--opening-plies N` | Adds N random opening moves to each FEN. | `0` |
| `--fens PATH` | Loads FEN4 opening positions from a file. | none |

Choose only one primary search limit: `--nodes`, `--depth`, `--movetime`, or
`--tc`. If none is supplied, `--nodes 10000` is used.

### Parameters and optimizer

| Flag | What it does | Default |
|---|---|---:|
| `--params PATH` | Loads a parameter manifest; repeatable. | required |
| `--group NAME`, `--tune-group NAME` | Tunes only parameters in a group; repeatable. | all trainable |
| `--fixed PATH` | Loads fixed option values from JSON; repeatable. | none |
| `--initial PATH` | Loads initial parameter values from JSON. | manifest defaults |
| `--iterations N` | Sets the total number of SPSA iterations. | `20` |
| `--pairs-per-iter N` | Sets opening pairs per exploration iteration. | `4` |
| `--learning-rate N` | Sets the SPSA learning-rate coefficient. | `1.5` |
| `--stability N` | Sets the learning-rate stability constant. | `4.0` |
| `--alpha N` | Sets the learning-rate decay exponent. | `0.602` |
| `--gamma N` | Sets the perturbation decay exponent. | `0.101` |
| `--seed N` | Sets the deterministic tuning/opening seed. | `20260608` |

Each pair contains two games with the tested `minus` and `plus` configurations
swapping teams.

### Phased tuning

| Flag | What it does | Default |
|---|---|---:|
| `--phased` | Enables exploration and refinement phases. | off |
| `--refine-pairs-multiplier N` | Multiplies `--pairs-per-iter` during refinement. | `2` |
| `--refine-start N` | Sets the first refinement iteration. | start of final third |
| `--refine-nodes N` | Uses a different node limit during refinement. | primary control |
| `--refine-depth N` | Uses a different depth during refinement. | primary control |
| `--refine-movetime N` | Uses different milliseconds per move during refinement. | primary control |
| `--refine-tc N` | Uses a different base clock in milliseconds during refinement. | primary control |
| `--refine-inc N` | Sets refinement clock increment; requires `--refine-tc`. | `0` |

The refinement controls are mutually exclusive and require `--phased`. Without
one, refinement keeps the primary search control. ETA calculations account for
phase pair counts, refinement controls, and observed refinement iteration time.

For `--iterations 20 --pairs-per-iter 4 --phased`, the default schedule is:

- Exploration: iterations 1-13 with 4 pairs per iteration.
- Refinement: iterations 14-20 with 8 pairs per iteration.

### Output and logging

| Flag | What it does | Default |
|---|---|---:|
| `--out-dir PATH` | Sets the run/checkpoint directory. | timestamped directory |
| `--resume`, `--no-resume` | Enables or disables checkpoint resume. | resume |
| `--fresh` | Deletes the selected output directory before starting. | off |
| `--verbose` | Prints phase, update, timing, and ETA details. | off |
| `--game-verbose` | Prints progress after each completed game. | off |
| `--save-games` | Appends full game records to `games.jsonl`. | off |

`--fresh` and `--no-resume` require an explicit `--out-dir`.

## Parameter manifest

The manifest is a JSON list, or an object containing a `parameters` list:

```json
{
  "parameters": [
    {
      "name": "MobilityWeight",
      "default": 100,
      "min": 0,
      "max": 300,
      "step": 1,
      "perturb": 10,
      "group": "evaluation",
      "train": true
    }
  ]
}
```

`name`, `default`, `min`, and `max` are required. `step` defaults to `1`, and
`perturb` defaults to `step`. Every parameter must be advertised by the engine
as a compatible UCI `spin` option.

List available manifest groups with:

```bash
PYTHONPATH=. python3 spsa/tune.py groups --params parameters.json
```

## Examples

Basic tune:

```bash
PYTHONPATH=. python3 spsa/tune.py spsa \
  --engine /path/to/engine \
  --params parameters.json \
  --iterations 100 \
  --pairs-per-iter 4 \
  --workers 8 \
  --movetime 250 \
  --out-dir tuning/my-run \
  --verbose
```

Phased tune with custom refinement:

```bash
PYTHONPATH=. python3 spsa/tune.py spsa \
  --engine /path/to/engine \
  --params parameters.json \
  --iterations 100 \
  --pairs-per-iter 4 \
  --workers 8 \
  --nodes 10000 \
  --phased \
  --refine-start 76 \
  --refine-pairs-multiplier 3 \
  --refine-nodes 30000 \
  --out-dir tuning/phased-run \
  --verbose
```

This runs iterations 1-75 with 4 pairs at 10,000 nodes, then iterations 76-100
with 12 pairs at 30,000 nodes.

## Run files

| File | Contents |
|---|---|
| `run.json` | Run fingerprint and immutable configuration. |
| `initial.json` | Initial values, fixed values, and trained parameter names. |
| `history.jsonl` | One optimizer result per completed iteration. |
| `latest.json` | Latest resumable checkpoint. |
| `latest.uci` | Latest values as UCI `setoption` commands. |
| `final.json` | Final values after the requested iterations. |
| `final.uci` | Final values as UCI `setoption` commands. |
| `games.jsonl` | Optional game records when `--save-games` is enabled. |

Rerun the same command to resume. Settings that affect the tune are
fingerprinted, and incompatible settings are rejected for an existing run.
