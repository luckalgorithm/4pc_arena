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


def validate_manifest_against_engine(
    args: argparse.Namespace,
    specs: list[dict[str, Any]],
    fixed: dict[str, int],
) -> str:
    base_options = match_runner.load_options(args.engine_options, args.engine_option)
    config = engine_config(
        args.engine,
        base_options,
        hash_mb=args.hash,
        threads=args.threads,
    )
    match_runner.validate_engine(config, "Engine")
    engine_name, uci_options = match_runner.probe_engine_options(config)
    errors: list[str] = []

    for spec in specs:
        name = str(spec["name"])
        option = uci_options.get(name)
        if option is None:
            errors.append(f"{name}: not advertised by the engine")
            continue
        if option.type != "spin":
            errors.append(
                f"{name}: manifest parameters must be UCI spin options, "
                f"but engine reports {option.type}"
            )
            continue
        if option.min is None or option.max is None:
            errors.append(f"{name}: engine did not advertise numeric bounds")
            continue
        if int(spec["min"]) < option.min:
            errors.append(
                f"{name}: manifest min {spec['min']} is below engine min {option.min}"
            )
        if int(spec["max"]) > option.max:
            errors.append(
                f"{name}: manifest max {spec['max']} is above engine max {option.max}"
            )
        if not option.min <= int(spec["default"]) <= option.max:
            errors.append(
                f"{name}: manifest default {spec['default']} is outside engine "
                f"range [{option.min}, {option.max}]"
            )

    for name, value in fixed.items():
        option = uci_options.get(name)
        if option is None:
            errors.append(f"fixed option {name}: not advertised by the engine")
        elif option.type == "spin" and (
            option.min is None
            or option.max is None
            or not option.min <= value <= option.max
        ):
            errors.append(
                f"fixed option {name}: value {value} is outside engine range "
                f"[{option.min}, {option.max}]"
            )

    if errors:
        details = "\n  ".join(errors)
        raise ValueError(
            f"Parameter manifest is incompatible with {engine_name}:\n  {details}"
        )
    return engine_name


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


def phase_schedule(
    iterations: int,
    pairs_per_iter: int,
    phased: bool,
    refine_pairs_multiplier: int = 2,
    refine_start: int | None = None,
) -> list[dict[str, Any]]:
    if not phased or iterations == 1:
        name = "exploration" if phased else "standard"
        return [
            {
                "name": name,
                "start": 1,
                "end": iterations,
                "iterations": iterations,
                "pairs_per_iter": pairs_per_iter,
            }
        ]

    exploration_pairs = pairs_per_iter
    refine_pairs = pairs_per_iter * refine_pairs_multiplier
    if refine_start is None:
        exploration_iterations = max(1, round(iterations * 2 / 3))
        exploration_iterations = min(exploration_iterations, iterations - 1)
    else:
        exploration_iterations = refine_start - 1
    refine_iterations = iterations - exploration_iterations
    return [
        {
            "name": "exploration",
            "start": 1,
            "end": exploration_iterations,
            "iterations": exploration_iterations,
            "pairs_per_iter": exploration_pairs,
        },
        {
            "name": "refine",
            "start": exploration_iterations + 1,
            "end": iterations,
            "iterations": refine_iterations,
            "pairs_per_iter": refine_pairs,
        },
    ]


def iteration_phase(
    schedule: list[dict[str, Any]],
    iteration: int,
) -> dict[str, Any]:
    for phase in schedule:
        if int(phase["start"]) <= iteration <= int(phase["end"]):
            return phase
    raise ValueError(f"No SPSA phase contains iteration {iteration}")


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
    control: dict[str, int | str],
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
        limit_kind=str(control["limit_kind"]),
        limit_value=int(control["limit_value"]),
        base_time_ms=int(control["tc"]),
        increment_ms=int(control["inc"]),
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
    pairs_per_iter: int,
    control: dict[str, int | str],
    minus: dict[str, int],
    plus: dict[str, int],
) -> tuple[float, dict[str, Any], list[dict[str, Any]]]:
    config = make_match_config(args, minus, plus, control)

    def show_game(
        record: dict[str, Any],
        records_by_id: dict[str, dict[str, Any]],
    ) -> None:
        if not args.game_verbose:
            return
        summary = match_runner.summarize(
            list(records_by_id.values()), pairs_per_iter * 2
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
        pairs=pairs_per_iter,
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
    payload: dict[str, Any] = {
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
    if args.phased:
        payload["phased"] = True
        payload["phase_schedule"] = phase_schedule(
            args.iterations,
            args.pairs_per_iter,
            True,
            args.refine_pairs_multiplier,
            args.refine_start,
        )
        if args.refine_limit_kind is not None:
            payload["refine_control"] = {
                "limit_kind": args.refine_limit_kind,
                "limit_value": args.refine_limit_value,
                "tc": args.refine_tc or 0,
                "inc": args.refine_inc,
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


def phase_control(
    args: argparse.Namespace,
    phase_name: str,
) -> dict[str, int | str]:
    if phase_name == "refine" and args.refine_limit_kind is not None:
        return {
            "limit_kind": args.refine_limit_kind,
            "limit_value": args.refine_limit_value,
            "tc": args.refine_tc or 0,
            "inc": args.refine_inc,
        }
    return {
        "limit_kind": args.limit_kind,
        "limit_value": args.limit_value,
        "tc": args.tc,
        "inc": args.inc,
    }


def control_text(control: dict[str, int | str]) -> str:
    if control["limit_kind"] == "clock":
        return f"tc {control['tc']}ms + {control['inc']}ms"
    return f"{control['limit_kind']} {control['limit_value']}"


def control_work(control: dict[str, int | str]) -> float | None:
    kind = control["limit_kind"]
    if kind in {"nodes", "movetime"}:
        return float(control["limit_value"])
    if kind == "clock":
        return float(control["tc"]) + 40.0 * float(control["inc"])
    return None


def control_ratio(
    source: dict[str, int | str],
    target: dict[str, int | str],
) -> float:
    if source["limit_kind"] != target["limit_kind"]:
        return 1.0
    source_work = control_work(source)
    target_work = control_work(target)
    if source_work is None or target_work is None or source_work <= 0:
        return 1.0
    return target_work / source_work


def estimate_eta_seconds(
    args: argparse.Namespace,
    schedule: list[dict[str, Any]],
    iteration: int,
    timing_history: list[dict[str, Any]],
) -> float | None:
    if not timing_history or iteration >= args.iterations:
        return None

    remaining = 0.0
    for future_iteration in range(iteration + 1, args.iterations + 1):
        phase = iteration_phase(schedule, future_iteration)
        target_control = phase_control(args, str(phase["name"]))
        exact = [
            sample
            for sample in timing_history
            if sample["phase"] == phase["name"]
            and sample["control"] == target_control
        ]
        samples = exact or timing_history
        per_pair = sum(
            float(sample["elapsed_seconds"])
            / int(sample["pairs_per_iter"])
            * control_ratio(sample["control"], target_control)
            for sample in samples
        ) / len(samples)
        remaining += per_pair * int(phase["pairs_per_iter"])
    return remaining


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
    phase_name: str,
    pairs_per_iter: int,
    control: dict[str, int | str],
    score: float,
    direction: float,
    summary: dict[str, Any],
    before: dict[str, int],
    after: dict[str, int],
    train_names: list[str],
    elapsed: float,
    schedule: list[dict[str, Any]],
    timing_history: list[dict[str, Any]],
) -> None:
    result = (
        f"+{summary['engine2_wins']} ={summary['draws']} "
        f"-{summary['engine1_wins']}"
    )
    if not args.verbose:
        print(
            f"[{iteration}/{args.iterations}] {phase_name} "
            f"({pairs_per_iter} pairs) | plus {result} | "
            f"{100 * score:.2f}% | direction {direction:+.3f}",
            flush=True,
        )
        return
    eta = "--"
    eta_seconds = estimate_eta_seconds(
        args,
        schedule,
        iteration,
        timing_history,
    )
    if eta_seconds is not None:
        eta = format_duration(eta_seconds)
    print(
        f"SPSA iteration {iteration}/{args.iterations} | {phase_name} | "
        f"{pairs_per_iter} pairs | {summary['games_completed']} games | "
        f"{control_text(control)}"
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
    engine_name = validate_manifest_against_engine(args, specs, fixed)
    print(
        f"Validated {len(specs)} parameter(s) against {engine_name}.",
        flush=True,
    )
    selected = [spec for spec in selected if str(spec["name"]) not in fixed]
    if not selected:
        raise ValueError("All selected parameters are fixed")
    train_names = [str(spec["name"]) for spec in selected]
    schedule = phase_schedule(
        args.iterations,
        args.pairs_per_iter,
        args.phased,
        args.refine_pairs_multiplier,
        args.refine_start,
    )

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
    base_control = phase_control(args, "exploration")
    timing_history = [
        {
            "phase": str(item.get("phase", "standard")),
            "pairs_per_iter": int(item.get("pairs_per_iter", args.pairs_per_iter)),
            "control": item.get("control", base_control),
            "elapsed_seconds": float(item["elapsed_seconds"]),
        }
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
        for phase in schedule:
            print(
                f"Phase {phase['name']}: iterations {phase['start']}-"
                f"{phase['end']}, {phase['pairs_per_iter']} pairs/iteration"
            )
        print(f"Output: {out_dir}", flush=True)

    for iteration in range(completed + 1, args.iterations + 1):
        started = time.monotonic()
        phase = iteration_phase(schedule, iteration)
        pairs_per_iter = int(phase["pairs_per_iter"])
        control = phase_control(args, str(phase["name"]))
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
            args, iteration, pairs_per_iter, control, minus, plus
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
        timing_history.append(
            {
                "phase": phase["name"],
                "pairs_per_iter": pairs_per_iter,
                "control": control,
                "elapsed_seconds": elapsed,
            }
        )
        payload = {
            "iteration": iteration,
            "phase": phase["name"],
            "pairs_per_iter": pairs_per_iter,
            "control": control,
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
            str(phase["name"]),
            pairs_per_iter,
            control,
            score,
            direction,
            summary,
            before,
            theta,
            train_names,
            elapsed,
            schedule,
            timing_history,
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
    spsa.add_argument(
        "--phased",
        action="store_true",
        help=(
            "use base-pair exploration iterations followed by higher-pair "
            "refinement iterations"
        ),
    )
    spsa.add_argument(
        "--refine-pairs-multiplier",
        type=int,
        default=2,
        metavar="N",
        help="multiply --pairs-per-iter by N during refinement (default: 2)",
    )
    spsa.add_argument(
        "--refine-start",
        type=int,
        metavar="ITERATION",
        help="first refinement iteration (default: start of final third)",
    )
    refine_limit = spsa.add_mutually_exclusive_group()
    refine_limit.add_argument("--refine-nodes", type=int)
    refine_limit.add_argument("--refine-depth", type=int)
    refine_limit.add_argument("--refine-movetime", type=int)
    refine_limit.add_argument(
        "--refine-tc",
        type=int,
        help="refinement base time in milliseconds",
    )
    spsa.add_argument(
        "--refine-inc",
        type=int,
        default=0,
        help="refinement clock increment in milliseconds",
    )
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
    if args.refine_tc is not None:
        args.refine_limit_kind = "clock"
        args.refine_limit_value = 0
    elif args.refine_movetime is not None:
        args.refine_limit_kind = "movetime"
        args.refine_limit_value = args.refine_movetime
    elif args.refine_depth is not None:
        args.refine_limit_kind = "depth"
        args.refine_limit_value = args.refine_depth
    elif args.refine_nodes is not None:
        args.refine_limit_kind = "nodes"
        args.refine_limit_value = args.refine_nodes
    else:
        args.refine_limit_kind = None
        args.refine_limit_value = 0
    if args.iterations < 1 or args.pairs_per_iter < 1:
        parser.error("--iterations and --pairs-per-iter must be at least 1")
    if args.refine_pairs_multiplier < 1:
        parser.error("--refine-pairs-multiplier must be at least 1")
    if not args.phased and (
        args.refine_pairs_multiplier != 2
        or args.refine_start is not None
        or args.refine_limit_kind is not None
        or args.refine_inc
    ):
        parser.error(
            "refinement options require --phased"
        )
    if args.refine_start is not None and not (
        2 <= args.refine_start <= args.iterations
    ):
        parser.error("--refine-start must be between 2 and --iterations")
    if args.workers < 1 or args.threads < 1:
        parser.error("--workers and --threads must be at least 1")
    if args.limit_kind == "clock" and args.tc <= 0:
        parser.error("--tc must be greater than 0")
    if args.limit_kind != "clock" and args.inc:
        parser.error("--inc requires --tc")
    if args.refine_limit_kind == "clock" and args.refine_tc <= 0:
        parser.error("--refine-tc must be greater than 0")
    if args.refine_limit_kind != "clock" and args.refine_inc:
        parser.error("--refine-inc requires --refine-tc")
    if args.refine_inc < 0:
        parser.error("--refine-inc must be non-negative")
    if (
        args.refine_limit_kind is not None
        and args.refine_limit_kind != "clock"
        and args.refine_limit_value <= 0
    ):
        parser.error("refinement search limit must be greater than 0")
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
