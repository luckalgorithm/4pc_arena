# 4 player chess match runner

Use `match.py` to run paired team matches between two four-player-chess UCI engines.

Python 3 is the only required runtime dependency.

| Flag | What it does | Default |
|---|---|---:|
| `-h`, `--help` | Shows command help. | |
| `--engine1`, `--e1` | Sets the Engine 1 executable. | required |
| `--engine2`, `--e2` | Sets the Engine 2 executable. | required |
| `--arbiter` | Uses an engine for strict legality and result checks. | none |
| `--engine1-options` | Loads Engine 1 UCI options from JSON. | none |
| `--engine2-options` | Loads Engine 2 UCI options from JSON. | none |
| `--arbiter-options` | Loads arbiter UCI options from JSON. | none |
| `--engine1-option NAME=VALUE` | Sets an Engine 1 UCI option; repeatable. | none |
| `--engine2-option NAME=VALUE` | Sets an Engine 2 UCI option; repeatable. | none |
| `--arbiter-option NAME=VALUE` | Sets an arbiter UCI option; repeatable. | none |
| `--threads1`, `--cores1` | Sets threads for each Engine 1 process. | `1` |
| `--threads2`, `--cores2` | Sets threads for each Engine 2 process. | `1` |
| `--hash1`, `--mem1` | Sets Engine 1 hash in MB. | `128` |
| `--hash2`, `--mem2` | Sets Engine 2 hash in MB. | `128` |
| `--arbiter-hash` | Sets arbiter hash in MB. | `16` |
| `--pairs N` | Runs N opening pairs, two games per pair. | `1` |
| `--games N` | Sets the total number of games; must be even. | |
| `--workers N` | Sets how many games run at once. | `1` |
| `--nodes N` | Sets nodes per move. | `10000`* |
| `--depth N` | Sets fixed search depth per move. | |
| `--movetime N`, `--fixed N` | Sets fixed milliseconds per move. | |
| `--tc N` | Sets base clock time in milliseconds. | |
| `--inc N` | Sets clock increment in milliseconds. | `0` |
| `--margin N` | Sets extra milliseconds allowed before a time loss. | `50` |
| `--timeout N` | Sets maximum seconds for an engine response. | `30.0` |
| `--max-plies N`, `--maxmoves N` | Draws after N total plies. | `1000` |
| `--opening-plies N` | Adds N random moves to each FEN opening. | `0` |
| `--fens PATH` | Loads FEN4 openings from a file. | none |
| `--seed N` | Sets the opening random seed. | `1` |
| `--out PATH` | Saves JSONL and enables resume. | none |
| `--pgn4 [PATH]` | Saves Chess.com-style PGN4. | none |
| `--moves`, `--pmoves` | Prints search information for every move. | off |
| `--quiet` | Prints only starting and final summaries. | off |
| `--continue-on-error`, `--continue` | Records runner errors as draws and continues. | off |
| `--resume`, `--no-resume` | Enables or disables resume for `--out`. | resume |
| `--fresh` | Deletes files associated with `--out` and restarts. | off |

## Basic Match

```bash
python3 match.py \
  --engine1 /path/to/engine1 \
  --engine2 /path/to/engine2 \
  --games 1000 \
  --workers 8 \
  --movetime 250
```

If `--fens` is provided, each opening is played twice with the engines on opposite teams. Results and Elo are shown from Engine 1's perspective.

Choose only one search limit: `--nodes`, `--depth`, `--movetime`, or `--tc`.
`--inc` requires `--tc`.

## Arbiter

An arbiter is normally unnecessary. Without one, both engines report moves and game results.

An explicit arbiter performs strict validation and must support `legalmoves` and `gameresult` commands.
