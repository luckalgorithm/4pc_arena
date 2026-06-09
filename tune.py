#!/usr/bin/env python3

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import match as match_runner


ROOT = Path(__file__).resolve().parent


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"File does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}") from exc


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(payload, sort_keys=True) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {path} line {line_number}") from exc
        if not isinstance(item, dict):
            raise ValueError(f"Invalid record in {path} line {line_number}")
        records.append(item)
    return records


def load_specs(paths: list[Path]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in paths:
        raw = read_json(path)
        if isinstance(raw, dict) and "parameters" in raw:
            raw = raw["parameters"]
        if not isinstance(raw, list):
            raise ValueError(f"Parameter manifest must be a list: {path}")
        for item in raw:
            if not isinstance(item, dict) or not item.get("name"):
                raise ValueError(f"Invalid parameter entry in {path}")
            spec = dict(item)
            name = str(spec["name"])
            if name in seen:
                raise ValueError(f"Duplicate parameter: {name}")
            for key in ("default", "min", "max"):
                if key not in spec:
                    raise ValueError(f"Parameter {name} is missing {key}")
            spec.setdefault("step", 1)
            spec.setdefault("perturb", max(1, int(spec["step"])))
            for key in ("default", "min", "max", "step", "perturb"):
                spec[key] = int(spec[key])
            if spec["min"] > spec["default"] or spec["default"] > spec["max"]:
                raise ValueError(f"Parameter {name} default is outside its bounds")
            if spec["step"] < 1 or spec["perturb"] < 1:
                raise ValueError(f"Parameter {name} step and perturb must be positive")
            seen.add(name)
            specs.append(spec)
    return specs


def spec_groups(spec: dict[str, Any]) -> set[str]:
    raw = spec.get("group", spec.get("groups", []))
    if isinstance(raw, str):
        return {raw}
    if isinstance(raw, list):
        return {str(group) for group in raw}
    if raw:
        raise ValueError(f"Invalid group on parameter {spec['name']}")
    return set()


def select_specs(
    specs: list[dict[str, Any]],
    requested_groups: list[str],
) -> list[dict[str, Any]]:
    available = {group for spec in specs for group in spec_groups(spec)}
    unknown = sorted(set(requested_groups) - available)
    if unknown:
        raise ValueError(f"Unknown tune group(s): {', '.join(unknown)}")
    selected = [
        spec
        for spec in specs
        if spec.get("train", True)
        and (
            not requested_groups
            or spec_groups(spec).intersection(requested_groups)
        )
    ]
    if not selected:
        raise ValueError("No trainable parameters were selected")
    return selected


def raw_values(path: Path) -> dict[str, int]:
    raw = read_json(path)
    if isinstance(raw, dict) and "values" in raw:
        raw = raw["values"]
    if not isinstance(raw, dict):
        raise ValueError(f"Values file must contain a JSON object: {path}")
    return {str(name): int(value) for name, value in raw.items()}


def load_fixed(paths: list[str]) -> dict[str, int]:
    values: dict[str, int] = {}
    for path in paths:
        values.update(raw_values(Path(path)))
    return values


def initial_values(
    specs: list[dict[str, Any]],
    initial_path: str,
) -> dict[str, int]:
    values = {str(spec["name"]): int(spec["default"]) for spec in specs}
    if initial_path:
        supplied = raw_values(Path(initial_path))
        unknown = sorted(set(supplied) - set(values))
        if unknown:
            raise ValueError(f"Initial values contain unknown parameters: {', '.join(unknown)}")
        values.update(supplied)
    return values


def clamp_float(value: float, spec: dict[str, Any]) -> float:
    return min(max(value, float(spec["min"])), float(spec["max"]))


def clamp_round(value: float, spec: dict[str, Any]) -> int:
    step = int(spec["step"])
    anchor = int(spec["default"])
    rounded = anchor + round((value - anchor) / step) * step
    return min(max(int(rounded), int(spec["min"])), int(spec["max"]))


def perturb_size(spec: dict[str, Any], schedule: float) -> int:
    return max(int(spec["step"]), int(round(int(spec["perturb"]) * schedule)))


def parameter_scale(spec: dict[str, Any]) -> float:
    return max(float(spec["step"]), float(spec["perturb"]), 1.0)


def timestamped_run_dir() -> Path:
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    return ROOT / "tuning" / f"spsa-{stamp}"


def engine_config(
    path: str,
    options: dict[str, int | bool | str],
    *,
    hash_mb: int,
    threads: int,
) -> match_runner.EngineConfig:
    return match_runner.EngineConfig(
        Path(path).resolve(),
        options,
        hash_mb,
        threads,
    )


def make_match_config(
    args: argparse.Namespace,
    minus: dict[str, int],
    plus: dict[str, int],
) -> match_runner.MatchConfig:
    base_options = match_runner.load_options(args.engine_options, args.engine_option)
    minus_options = dict(base_options)
    minus_options.update(minus)
    plus_options = dict(base_options)
    plus_options.update(plus)
    arbiter = None
    if args.arbiter:
        arbiter = engine_config(
            args.arbiter,
            match_runner.load_options(args.arbiter_options, args.arbiter_option),
            hash_mb=args.arbiter_hash,
            threads=1,
        )
    return match_runner.MatchConfig(
        engine1=engine_config(
            args.engine, minus_options, hash_mb=args.hash, threads=args.threads
        ),
        engine2=engine_config(
            args.engine, plus_options, hash_mb=args.hash, threads=args.threads
        ),
        arbiter=arbiter,
        limit_kind=args.limit_kind,
        limit_value=args.limit_value,
        base_time_ms=args.tc,
        increment_ms=args.inc,
        timeout=args.timeout,
        margin_ms=args.margin,
        max_plies=args.max_plies,
        opening_plies=args.opening_plies,
        fens=match_runner.load_fens(args.fens),
        workers=args.workers,
        show_moves=False,
        out=None,
    )


def save_iteration_games(
    path: Path,
    iteration: int,
    records: list[dict[str, Any]],
) -> None:
    for record in records:
        item = dict(record)
        item["iteration"] = iteration
        item["iteration_game_id"] = item["game_id"]
        item["game_id"] = f"iter{iteration:04d}:{item['game_id']}"
        append_jsonl(path, item)


def run_iteration(
    args: argparse.Namespace,
    iteration: int,
    minus: dict[str, int],
    plus: dict[str, int],
) -> tuple[float, dict[str, Any], list[dict[str, Any]]]:
    config = make_match_config(args, minus, plus)

    def show_game(
        record: dict[str, Any],
        records_by_id: dict[str, dict[str, Any]],
    ) -> None:
        if not args.game_verbose:
            return
        summary = match_runner.summarize(
            list(records_by_id.values()), args.pairs_per_iter * 2
        )
        print(
            f"  game {summary['games_completed']}/{summary['games_expected']}: "
            f"plus {100 * summary['engine2_score']:.2f}% "
            f"(+{summary['engine2_wins']} ={summary['draws']} "
            f"-{summary['engine1_wins']})",
            flush=True,
        )

    result = match_runner.play_match(
        config,
        pairs=args.pairs_per_iter,
        seed=args.seed * 10_000 + iteration,
        on_record=show_game,
    )
    if result.interrupted:
        raise KeyboardInterrupt
    return float(result.summary["engine2_score"]), result.summary, result.records


def run_fingerprint(
    args: argparse.Namespace,
    specs: list[dict[str, Any]],
    train_names: list[str],
    fixed: dict[str, int],
) -> dict[str, Any]:
    payload = {
        "format": 1,
        "optimizer": "normalized_spsa",
        "engine": str(Path(args.engine).resolve()),
        "engine_options": match_runner.load_options(
            args.engine_options, args.engine_option
        ),
        "arbiter": str(Path(args.arbiter).resolve()) if args.arbiter else "",
        "arbiter_options": match_runner.load_options(
            args.arbiter_options, args.arbiter_option
        ),
        "specs": specs,
        "train_names": train_names,
        "fixed": fixed,
        "initial": str(Path(args.initial).resolve()) if args.initial else "",
        "groups": args.group,
        "pairs_per_iter": args.pairs_per_iter,
        "limit_kind": args.limit_kind,
        "limit_value": args.limit_value,
        "tc": args.tc,
        "inc": args.inc,
        "max_plies": args.max_plies,
        "opening_plies": args.opening_plies,
        "fens": str(Path(args.fens).resolve()) if args.fens else "",
        "hash": args.hash,
        "threads": args.threads,
        "timeout": args.timeout,
        "margin": args.margin,
        "learning_rate": args.learning_rate,
        "stability": args.stability,
        "alpha": args.alpha,
        "gamma": args.gamma,
        "seed": args.seed,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return {
        "fingerprint": hashlib.sha256(encoded).hexdigest(),
        "config": payload,
    }


def write_uci(path: Path, values: dict[str, int]) -> None:
    lines = [
        "# Generated by tune.py",
        *[
            f"setoption name {name} value {value}"
            for name, value in sorted(values.items())
        ],
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_duration(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def control_text(args: argparse.Namespace) -> str:
    if args.limit_kind == "clock":
        return f"tc {args.tc}ms + {args.inc}ms"
    return f"{args.limit_kind} {args.limit_value}"


def changed_text(
    before: dict[str, int],
    after: dict[str, int],
    names: list[str],
) -> str:
    changes = [
        f"{name} {before[name]}->{after[name]}"
        for name in names
        if before[name] != after[name]
    ]
    if not changes:
        return "none"
    if len(changes) > 8:
        return ", ".join(changes[:8]) + f", +{len(changes) - 8} more"
    return ", ".join(changes)


def print_iteration(
    args: argparse.Namespace,
    iteration: int,
    score: float,
    direction: float,
    summary: dict[str, Any],
    before: dict[str, int],
    after: dict[str, int],
    train_names: list[str],
    elapsed: float,
    elapsed_history: list[float],
) -> None:
    result = (
        f"+{summary['engine2_wins']} ={summary['draws']} "
        f"-{summary['engine1_wins']}"
    )
    if not args.verbose:
        print(
            f"[{iteration}/{args.iterations}] plus {result} | "
            f"{100 * score:.2f}% | direction {direction:+.3f}",
            flush=True,
        )
        return
    eta = "--"
    if args.limit_kind in {"movetime", "clock"} and elapsed_history:
        average = sum(elapsed_history) / len(elapsed_history)
        eta = format_duration(average * (args.iterations - iteration))
    print(
        f"SPSA iteration {iteration}/{args.iterations} | "
        f"{summary['games_completed']} games | {control_text(args)}"
    )
    print(f"  Plus: {result} | score {score:.3f} | direction {direction:+.3f}")
    print(f"  Update: {changed_text(before, after, train_names)}")
    print(f"  Time: {format_duration(elapsed)} | ETA: {eta}", flush=True)


def command_groups(args: argparse.Namespace) -> None:
    specs = load_specs([Path(path) for path in args.params])
    groups: dict[str, list[str]] = {}
    ungrouped: list[str] = []
    for spec in specs:
        names = spec_groups(spec)
        if not names:
            ungrouped.append(str(spec["name"]))
        for group in names:
            groups.setdefault(group, []).append(str(spec["name"]))
    for group in sorted(groups):
        print(f"{group}:")
        for name in groups[group]:
            print(f"  {name}")
    if ungrouped:
        print("ungrouped:")
        for name in ungrouped:
            print(f"  {name}")


def command_spsa(args: argparse.Namespace) -> None:
    spec_paths = [Path(path) for path in args.params]
    specs = load_specs(spec_paths)
    selected = select_specs(specs, args.group)
    fixed = load_fixed(args.fixed)
    selected = [spec for spec in selected if str(spec["name"]) not in fixed]
    if not selected:
        raise ValueError("All selected parameters are fixed")
    train_names = [str(spec["name"]) for spec in selected]

    out_dir = Path(args.out_dir).resolve() if args.out_dir else timestamped_run_dir()
    if args.fresh and out_dir.exists():
        shutil.rmtree(out_dir)
    elif out_dir.exists() and not args.resume and any(out_dir.iterdir()):
        raise ValueError(f"Output directory is not empty: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    run = run_fingerprint(args, specs, train_names, fixed)
    run_path = out_dir / "run.json"
    if run_path.is_file():
        existing = read_json(run_path)
        if existing.get("fingerprint") != run["fingerprint"]:
            raise ValueError(
                f"Existing run settings differ: {run_path}. "
                "Use the original settings, another --out-dir, or --fresh."
            )
    else:
        write_json(run_path, run)

    history_path = out_dir / "history.jsonl"
    history = load_jsonl(history_path) if args.resume else []
    completed = max((int(item["iteration"]) for item in history), default=0)
    elapsed_history = [
        float(item["elapsed_seconds"])
        for item in history
        if isinstance(item.get("elapsed_seconds"), (int, float))
    ]

    if completed:
        latest_path = out_dir / "latest.json"
        if not latest_path.is_file():
            raise ValueError(f"Missing resumable checkpoint: {latest_path}")
        latest = read_json(latest_path)
        theta = {str(name): int(value) for name, value in latest["tuned_values"].items()}
        theta_float = {
            str(name): float(value) for name, value in latest["theta_float"].items()
        }
        print(f"Resuming after iteration {completed}.", flush=True)
    else:
        theta = initial_values(specs, args.initial)
        theta_float = {
            str(spec["name"]): float(theta[str(spec["name"])])
            for spec in selected
        }
        write_json(
            out_dir / "initial.json",
            {
                "values": {**theta, **fixed},
                "fixed": fixed,
                "train_names": train_names,
            },
        )

    if args.verbose:
        print(f"Engine: {Path(args.engine).resolve()}")
        print(f"Training {len(train_names)} parameter(s): {', '.join(train_names)}")
        print(f"Output: {out_dir}", flush=True)

    for iteration in range(completed + 1, args.iterations + 1):
        started = time.monotonic()
        rng = random.Random(args.seed * 1_000_003 + iteration)
        ck = 1.0 / (iteration**args.gamma)
        ak = args.learning_rate / ((iteration + args.stability) ** args.alpha)
        plus = {**theta, **fixed}
        minus = {**theta, **fixed}
        delta: dict[str, int] = {}
        actual_delta: dict[str, int] = {}
        perturb: dict[str, int] = {}
        scales: dict[str, float] = {}

        for spec in selected:
            name = str(spec["name"])
            amount = perturb_size(spec, ck)
            sign = rng.choice((-1, 1))
            plus[name] = clamp_round(theta_float[name] + sign * amount, spec)
            minus[name] = clamp_round(theta_float[name] - sign * amount, spec)
            delta[name] = sign
            actual_delta[name] = plus[name] - minus[name]
            perturb[name] = amount
            scales[name] = parameter_scale(spec)

        score, summary, records = run_iteration(
            args, iteration, minus, plus
        )
        direction = 2.0 * (score - 0.5)
        before = dict(theta)
        gradients: dict[str, float | None] = {}
        updates: dict[str, float] = {}
        for spec in selected:
            name = str(spec["name"])
            tested_delta = actual_delta[name]
            if tested_delta == 0:
                gradients[name] = None
                updates[name] = 0.0
                continue
            scale = scales[name]
            gradient = direction * scale / tested_delta
            update = ak * gradient * scale
            gradients[name] = gradient
            updates[name] = update
            theta_float[name] = clamp_float(theta_float[name] + update, spec)
            theta[name] = clamp_round(theta_float[name], spec)

        elapsed = time.monotonic() - started
        elapsed_history.append(elapsed)
        payload = {
            "iteration": iteration,
            "score_plus": score,
            "direction": direction,
            "learning_rate": ak,
            "elapsed_seconds": elapsed,
            "theta": theta,
            "theta_float": theta_float,
            "plus": plus,
            "minus": minus,
            "delta": delta,
            "perturb": perturb,
            "actual_delta": actual_delta,
            "gradient": gradients,
            "update": updates,
            "summary": summary,
        }
        append_jsonl(history_path, payload)
        checkpoint = {
            "values": {**theta, **fixed},
            "tuned_values": theta,
            "theta_float": theta_float,
            "fixed": fixed,
            "iteration": iteration,
            "train_names": train_names,
        }
        write_json(out_dir / "latest.json", checkpoint)
        write_uci(out_dir / "latest.uci", checkpoint["values"])
        if args.save_games:
            save_iteration_games(out_dir / "games.jsonl", iteration, records)
        print_iteration(
            args,
            iteration,
            score,
            direction,
            summary,
            before,
            theta,
            train_names,
            elapsed,
            elapsed_history,
        )

    final = {
        "values": {**theta, **fixed},
        "tuned_values": theta,
        "theta_float": theta_float,
        "fixed": fixed,
        "iterations": max(completed, args.iterations),
        "train_names": train_names,
    }
    write_json(out_dir / "final.json", final)
    write_uci(out_dir / "final.uci", final["values"])
    print(f"Final values: {out_dir / 'final.json'}", flush=True)


def add_match_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--engine", required=True, help="UCI engine executable")
    parser.add_argument("--engine-options", default="", help="base UCI options JSON")
    parser.add_argument("--engine-option", action="append", default=[], metavar="NAME=VALUE")
    parser.add_argument("--arbiter", default="", help="optional dedicated rules engine")
    parser.add_argument("--arbiter-options", default="", help="arbiter UCI options JSON")
    parser.add_argument("--arbiter-option", action="append", default=[], metavar="NAME=VALUE")
    parser.add_argument("--arbiter-hash", type=int, default=16)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--hash", type=int, default=128)
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(4, os.cpu_count() or 1)),
    )
    limit = parser.add_mutually_exclusive_group()
    limit.add_argument("--nodes", type=int)
    limit.add_argument("--depth", type=int)
    limit.add_argument("--movetime", type=int)
    limit.add_argument("--tc", type=int, help="base time in milliseconds")
    parser.add_argument("--inc", type=int, default=0)
    parser.add_argument("--margin", type=int, default=50)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--max-plies", "--maxmoves", type=int, default=1000)
    parser.add_argument("--opening-plies", type=int, default=0)
    parser.add_argument("--fens", default="")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SPSA tuning for 4PC UCI engines"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    groups = commands.add_parser("groups", help="list groups in parameter manifests")
    groups.add_argument("--params", action="append", required=True)
    groups.set_defaults(func=command_groups)

    spsa = commands.add_parser("spsa", help="run an SPSA tune")
    add_match_args(spsa)
    spsa.add_argument("--params", action="append", required=True)
    spsa.add_argument("--group", "--tune-group", action="append", default=[])
    spsa.add_argument("--fixed", action="append", default=[])
    spsa.add_argument("--initial", default="")
    spsa.add_argument("--iterations", type=int, default=20)
    spsa.add_argument("--pairs-per-iter", type=int, default=4)
    spsa.add_argument("--learning-rate", type=float, default=1.5)
    spsa.add_argument("--stability", type=float, default=4.0)
    spsa.add_argument("--alpha", type=float, default=0.602)
    spsa.add_argument("--gamma", type=float, default=0.101)
    spsa.add_argument("--seed", type=int, default=20260608)
    spsa.add_argument("--out-dir", default="")
    spsa.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    spsa.add_argument("--fresh", action="store_true")
    spsa.add_argument("--verbose", action="store_true")
    spsa.add_argument("--game-verbose", action="store_true")
    spsa.add_argument(
        "--save-games",
        action="store_true",
        help="append game records to one games.jsonl file",
    )
    spsa.set_defaults(func=command_spsa)
    return parser


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.command != "spsa":
        return
    if args.tc is not None:
        args.limit_kind = "clock"
        args.limit_value = 0
    elif args.movetime is not None:
        args.limit_kind = "movetime"
        args.limit_value = args.movetime
        args.tc = 0
    elif args.depth is not None:
        args.limit_kind = "depth"
        args.limit_value = args.depth
        args.tc = 0
    else:
        args.limit_kind = "nodes"
        args.limit_value = args.nodes if args.nodes is not None else 10000
        args.tc = 0
    if args.iterations < 1 or args.pairs_per_iter < 1:
        parser.error("--iterations and --pairs-per-iter must be at least 1")
    if args.workers < 1 or args.threads < 1:
        parser.error("--workers and --threads must be at least 1")
    if args.limit_kind == "clock" and args.tc <= 0:
        parser.error("--tc must be greater than 0")
    if args.limit_kind != "clock" and args.inc:
        parser.error("--inc requires --tc")
    if args.learning_rate <= 0 or args.stability < 0:
        parser.error("--learning-rate must be positive and --stability non-negative")
    if args.alpha <= 0 or args.gamma <= 0:
        parser.error("--alpha and --gamma must be positive")
    if not args.arbiter and (args.arbiter_options or args.arbiter_option):
        parser.error("--arbiter-options and --arbiter-option require --arbiter")
    if (args.fresh or not args.resume) and not args.out_dir:
        parser.error("--fresh and --no-resume require --out-dir")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args, parser)
    try:
        args.func(args)
    except KeyboardInterrupt:
        print(
            "Tuning stopped. Completed iterations are saved; "
            "rerun the same command to resume.",
            file=sys.stderr,
        )
        return 130
    except (match_runner.EngineError, TimeoutError, OSError, ValueError) as exc:
        print(f"tune: error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
