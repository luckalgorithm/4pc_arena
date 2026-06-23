#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from match import EngineConfig, MatchConfig, load_fens, load_options, play_match


DEFAULT_ALPHA = 0.602
DEFAULT_GAMMA = 0.101


@dataclass
class Parameter:
    name: str
    theta: float
    start: float
    minimum: float
    maximum: float
    c: float
    c_end: float
    a: float
    a_end: float
    r_end: float


@dataclass
class Step:
    parameter: str
    flip: int
    c: float
    r: float
    plus: float
    minus: float


def finite_float(value: str, *, field: str, line: int | None = None) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        location = f"line {line} {field}" if line is not None else field
        raise ValueError(f"Invalid SPSA {location}: expected a number") from exc
    if not math.isfinite(number):
        location = f"line {line} {field}" if line is not None else field
        raise ValueError(f"Invalid SPSA {location}: expected a finite number")
    return number


def clip(value: float, minimum: float, maximum: float) -> float:
    return min(max(value, minimum), maximum)


def option_value(value: float) -> int | str:
    rounded = round(value)
    if abs(value - rounded) < 1.0e-9:
        return int(rounded)
    return f"{value:.10g}"


def parse_parameters(
    path: Path,
    *,
    iterations: int,
    a: float,
    alpha: float,
    gamma: float,
) -> list[Parameter]:
    params: list[Parameter] = []
    raw = path.read_text(encoding="utf-8")
    for line_number, raw_line in enumerate(raw.splitlines(), 1):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        chunks = [chunk.strip() for chunk in line.split(",")]
        if len(chunks) != 6:
            raise ValueError(
                f"Invalid SPSA line {line_number}: expected "
                "name,start,min,max,c_end,r_end"
            )
        name = chunks[0]
        if not name:
            raise ValueError(f"Invalid SPSA line {line_number}: empty name")
        start = finite_float(chunks[1], field="start", line=line_number)
        minimum = finite_float(chunks[2], field="min", line=line_number)
        maximum = finite_float(chunks[3], field="max", line=line_number)
        c_end = finite_float(chunks[4], field="c_end", line=line_number)
        r_end = finite_float(chunks[5], field="r_end", line=line_number)
        if minimum > maximum:
            raise ValueError(f"Invalid SPSA line {line_number}: min > max")
        if c_end <= 0:
            raise ValueError(f"Invalid SPSA line {line_number}: c_end must be > 0")
        if r_end < 0:
            raise ValueError(f"Invalid SPSA line {line_number}: r_end must be >= 0")
        theta = clip(start, minimum, maximum)
        c = c_end * iterations**gamma
        a_end = r_end * c_end**2
        params.append(
            Parameter(
                name=name,
                theta=theta,
                start=start,
                minimum=minimum,
                maximum=maximum,
                c=c,
                c_end=c_end,
                a=a_end * (a + iterations) ** alpha,
                a_end=a_end,
                r_end=r_end,
            )
        )
    if not params:
        raise ValueError(f"No SPSA parameters found: {path}")
    return params


def load_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def restore_parameters(params: list[Parameter], state: dict[str, Any]) -> int:
    by_name = {param.name: param for param in params}
    for saved in state.get("parameters", []):
        name = saved.get("name")
        if name in by_name:
            by_name[name].theta = clip(
                float(saved["theta"]),
                by_name[name].minimum,
                by_name[name].maximum,
            )
    return int(state.get("iteration", 0))


def make_steps(
    params: list[Parameter],
    *,
    iteration: int,
    a: float,
    alpha: float,
    gamma: float,
    rng: random.Random,
) -> list[Step]:
    iter_local = iteration + 1
    steps: list[Step] = []
    for param in params:
        flip = rng.choice((-1, 1))
        c = param.c / iter_local**gamma
        r = param.a / (a + iter_local) ** alpha / c**2
        steps.append(
            Step(
                parameter=param.name,
                flip=flip,
                c=c,
                r=r,
                plus=clip(param.theta + c * flip, param.minimum, param.maximum),
                minus=clip(param.theta - c * flip, param.minimum, param.maximum),
            )
        )
    return steps


def options_from_steps(
    base_options: dict[str, int | bool | str],
    steps: list[Step],
    *,
    plus: bool,
) -> dict[str, int | bool | str]:
    options = dict(base_options)
    for step in steps:
        value = step.plus if plus else step.minus
        options[step.parameter] = option_value(value)
    return options


def update_parameters(params: list[Parameter], steps: list[Step], result: int) -> None:
    by_name = {param.name: param for param in params}
    for step in steps:
        param = by_name[step.parameter]
        delta = step.r * step.c * result * step.flip
        param.theta = clip(param.theta + delta, param.minimum, param.maximum)


def parameter_rows(
    params: list[Parameter],
    *,
    iteration: int,
    a: float,
    alpha: float,
    gamma: float,
) -> list[str]:
    iter_local = iteration + 1
    rows = []
    for param in params:
        c = param.c / iter_local**gamma
        r = param.a / (a + iter_local) ** alpha / c**2
        rows.append(
            f"{param.name},{param.theta:.6g},{param.minimum:.6g},{param.maximum:.6g},"
            f"{c:.6g},{r:.6g}"
        )
    return rows


def cxx_string(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')


def rounded_theta(param: dict[str, Any]) -> int:
    value = round(float(param["theta"]))
    return int(clip(value, float(param["minimum"]), float(param["maximum"])))


def export_state(args: argparse.Namespace) -> int:
    state = load_state(Path(args.state))
    if not state:
        raise ValueError(f"No state found: {args.state}")
    params = state.get("parameters")
    if not isinstance(params, list) or not params:
        raise ValueError(f"State has no parameters: {args.state}")

    params_lines = []
    cxx_lines = [
        "void Tune::read_results() {",
        f"    // Exported from {args.state} at iteration {state.get('iteration', 0)}",
    ]
    for param in params:
        value = rounded_theta(param)
        name = str(param["name"])
        params_lines.append(
            f"{name},{value},{int(float(param['minimum']))},"
            f"{int(float(param['maximum']))},{float(param['c_end']):.6g},"
            f"{float(param['r_end']):.6g}"
        )
        cxx_lines.append(f'    TuneResults["{cxx_string(name)}"] = {value};')
    cxx_lines.append("}")

    if args.params_out:
        write_text(Path(args.params_out), "\n".join(params_lines) + "\n")
    if args.cxx_out:
        write_text(Path(args.cxx_out), "\n".join(cxx_lines) + "\n")
    if not args.params_out and not args.cxx_out:
        print("\n".join(cxx_lines))
    else:
        print(
            f"Exported {len(params)} params from iteration "
            f"{state.get('iteration', 0)}"
        )
    return 0


def build_match_config(
    args: argparse.Namespace,
    *,
    plus_options: dict[str, int | bool | str],
    minus_options: dict[str, int | bool | str],
) -> MatchConfig:
    engine = Path(args.engine).resolve()
    arbiter = Path(args.arbiter).resolve() if args.arbiter else None
    return MatchConfig(
        engine1=EngineConfig(engine, plus_options, args.hash, args.threads),
        engine2=EngineConfig(engine, minus_options, args.hash, args.threads),
        arbiter=EngineConfig(arbiter, {}, args.arbiter_hash, 1) if arbiter else None,
        limit_kind=args.limit_kind,
        limit_value=args.limit_value,
        base_time_ms=args.tc,
        increment_ms=args.inc,
        timeout=args.timeout,
        margin_ms=args.margin,
        max_plies=args.max_plies,
        opening_plies=args.opening_plies,
        fens=load_fens(args.fens),
        workers=args.workers,
        show_moves=False,
        out=None,
    )


def result_from_summary(summary: dict[str, Any]) -> int:
    return int(summary["engine1_wins"]) - int(summary["engine2_wins"])


def summarize_step(summary: dict[str, Any]) -> str:
    return (
        f"WDL {summary['engine1_wins']}/"
        f"{summary['draws']}/{summary['engine2_wins']} "
        f"score {100.0 * summary['engine1_score']:.2f}%"
    )


def run_spsa(args: argparse.Namespace) -> int:
    if args.iterations < 1:
        raise ValueError("--iterations must be at least 1")
    if args.pairs < 1:
        raise ValueError("--pairs must be at least 1")
    if args.a is None:
        args.a = 0.1 * args.iterations
    if args.a < 0:
        raise ValueError("--A must be >= 0")

    base_options = load_options(args.options, args.option)
    params = parse_parameters(
        Path(args.params),
        iterations=args.iterations,
        a=args.a,
        alpha=args.alpha,
        gamma=args.gamma,
    )
    state_path = Path(args.state)
    state = load_state(state_path) if args.resume else {}
    start_iteration = restore_parameters(params, state)
    if start_iteration >= args.iterations:
        print("SPSA already complete.")
        return 0

    history = list(state.get("history", []))
    rng = random.Random(args.seed)
    for _ in range(len(history)):
        for _param in params:
            rng.choice((-1, 1))

    started_at = state.get(
        "started_at",
        time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )

    iteration = start_iteration
    while iteration < args.iterations:
        batch_pairs = min(args.pairs, args.iterations - iteration)
        steps = make_steps(
            params,
            iteration=iteration,
            a=args.a,
            alpha=args.alpha,
            gamma=args.gamma,
            rng=rng,
        )
        plus_options = options_from_steps(base_options, steps, plus=True)
        minus_options = options_from_steps(base_options, steps, plus=False)
        config = build_match_config(
            args,
            plus_options=plus_options,
            minus_options=minus_options,
        )
        match_result = play_match(
            config,
            pairs=batch_pairs,
            seed=args.seed + iteration,
            continue_on_error=args.continue_on_error,
        )
        summary = match_result.summary
        result = result_from_summary(summary)
        update_parameters(params, steps, result)
        iteration += batch_pairs

        entry = {
            "iteration": iteration,
            "pairs": batch_pairs,
            "result": result,
            "summary": summary,
            "steps": [asdict(step) for step in steps],
            "parameters": [asdict(param) for param in params],
        }
        history.append(entry)
        state = {
            "format": 1,
            "started_at": started_at,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "iteration": iteration,
            "iterations": args.iterations,
            "pairs": args.pairs,
            "seed": args.seed,
            "alpha": args.alpha,
            "gamma": args.gamma,
            "A": args.a,
            "parameters": [asdict(param) for param in params],
            "history": history,
        }
        write_state(state_path, state)

        if iteration % args.progress_interval == 0 or iteration == args.iterations:
            print(
                f"[{iteration}/{args.iterations}] {summarize_step(summary)} "
                f"gradient {result:+d}",
                flush=True,
            )
        if args.param_interval and (
            iteration % args.param_interval == 0 or iteration == args.iterations
        ):
            for row in parameter_rows(
                params,
                iteration=iteration,
                a=args.a,
                alpha=args.alpha,
                gamma=args.gamma,
            ):
                print(row, flush=True)

    return 0


def build_tune_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Stockfish-style SPSA tuning against a 4PC UCI engine"
    )
    parser.add_argument("--engine", required=True, help="stockfish_4pc executable")
    parser.add_argument("--params", required=True, help="SPSA parameter file")
    parser.add_argument("--state", default="spsa_state.json", help="resumable state file")
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument(
        "--pairs",
        type=int,
        default=1,
        help="paired openings per SPSA update",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--A", dest="a", type=float)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    parser.add_argument("--options", default="", help="base engine options JSON")
    parser.add_argument("--option", action="append", default=[], metavar="NAME=VALUE")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--hash", type=int, default=128)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--arbiter", default="")
    parser.add_argument("--arbiter-hash", type=int, default=16)
    limit = parser.add_mutually_exclusive_group()
    limit.add_argument("--nodes", type=int)
    limit.add_argument("--depth", type=int)
    limit.add_argument("--movetime", type=int)
    limit.add_argument("--tc", type=int, help="base time in milliseconds")
    parser.add_argument("--inc", type=int, default=0)
    parser.add_argument("--margin", type=int, default=50)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--max-plies", type=int, default=1000)
    parser.add_argument("--opening-plies", type=int, default=0)
    parser.add_argument("--fens", default="")
    parser.add_argument(
        "--allow-startpos",
        action="store_true",
        help="allow tuning every pair from startpos; useful only for smoke tests",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=1,
        help="print one summary line every N SPSA iterations",
    )
    parser.add_argument(
        "--param-interval",
        type=int,
        default=0,
        help="also print all parameter rows every N SPSA iterations; 0 disables",
    )
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser


def build_export_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export tuned SPSA values from a state file"
    )
    parser.add_argument("--state", required=True, help="SPSA state JSON")
    parser.add_argument(
        "--cxx-out",
        default="",
        help="write Tune::read_results() replacement snippet",
    )
    parser.add_argument(
        "--params-out",
        default="",
        help="write rounded params file for continuing/testing",
    )
    return parser


def normalize_args(args: argparse.Namespace) -> None:
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
    if args.limit_kind == "clock" and args.tc <= 0:
        raise ValueError("--tc must be greater than 0")
    if args.limit_kind != "clock" and args.inc:
        raise ValueError("--inc requires --tc")
    if args.threads < 1 or args.workers < 1:
        raise ValueError("--threads and --workers must be at least 1")
    if not args.fens and args.opening_plies <= 0 and not args.allow_startpos:
        raise ValueError(
            "SPSA needs opening diversity: pass --fens PATH, or use "
            "--allow-startpos only for smoke tests"
        )
    if args.opening_plies > 0 and not args.arbiter:
        raise ValueError(
            "--opening-plies requires --arbiter with legalmoves support for this engine"
        )
    if args.progress_interval < 1:
        raise ValueError("--progress-interval must be at least 1")
    if args.param_interval < 0:
        raise ValueError("--param-interval must be >= 0")


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "export":
        parser = build_export_parser()
        args = parser.parse_args(sys.argv[2:])
        try:
            return export_state(args)
        except (OSError, ValueError) as exc:
            print(f"spsa export: error: {exc}", file=sys.stderr)
            return 2

    parser = build_tune_parser()
    args = parser.parse_args()
    try:
        normalize_args(args)
        return run_spsa(args)
    except KeyboardInterrupt:
        print(
            "Interrupted; state is saved after each completed SPSA update.",
            file=sys.stderr,
        )
        return 130
    except (OSError, ValueError, TimeoutError) as exc:
        print(f"spsa: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
