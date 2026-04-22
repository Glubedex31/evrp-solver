import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

from main_4 import (
    BatchSummary,
    ContinuousPartialPolicy,
    DEFAULT_FIXED_LEVELS,
    FixedPartialPolicy,
    FullChargePolicy,
    ObjectiveValue,
    PolicySpec,
    RunConfig,
    RunResult,
    default_policy_specs,
    list_schneider_instances,
    objective_string,
    plot_solution,
    read_schneider_instance,
    solve_run,
)


def _batch_manifest(
    preset_name: str,
    output_dir: Path,
    include_validation: bool,
    configs: Sequence[RunConfig],
) -> Dict[str, object]:
    unique_policies = []
    seen_policy_keys = set()
    for config in configs:
        key = config.policy_spec.key
        if key in seen_policy_keys:
            continue
        seen_policy_keys.add(key)
        unique_policies.append(
            {
                "name": config.policy_spec.name,
                "label": config.policy_spec.label,
                "fixed_levels": list(config.policy_spec.fixed_levels),
            }
        )

    return {
        "preset_name": preset_name,
        "output_dir": str(output_dir),
        "include_validation": include_validation,
        "run_count": len(configs),
        "instance_count": len({Path(config.instance_path).name for config in configs}),
        "seeds": sorted({config.seed for config in configs}),
        "time_budget_seconds": sorted(
            {config.time_budget_seconds for config in configs}
        ),
        "remove_fractions": sorted({config.remove_fraction for config in configs}),
        "policies": unique_policies,
    }


def preset_instance_paths(preset: str, data_dir: str = "data_schneider") -> List[str]:
    paths = list_schneider_instances(data_dir)
    names = {Path(path).name: path for path in paths}

    if preset == "sanity":
        return [path for path in paths if any(tag in Path(path).stem for tag in ("C5", "C10", "C15"))]

    if preset == "pilot":
        pilot_names = [
            "c101C5.txt",
            "c101C10.txt",
            "c101_21.txt",
            "r101_21.txt",
            "r102C10.txt",
            "rc101_21.txt",
            "rc108C5.txt",
        ]
        return [names[name] for name in pilot_names if name in names]

    if preset == "paper_subset":
        subset_names = [
            "c101C10.txt",
            "c101_21.txt",
            "r102C10.txt",
            "rc101_21.txt",
        ]
        return [names[name] for name in subset_names if name in names]

    if preset == "validation":
        validation_names = ["c101C5.txt", "r102C10.txt", "rc108C5.txt"]
        return [names[name] for name in validation_names if name in names]

    if preset == "paper_full":
        return paths

    raise ValueError(f"Unknown preset: {preset}")


def build_run_configs(
    preset: str,
    time_budget_seconds: float,
    remove_fraction: float,
    seeds: Sequence[int],
    data_dir: str = "data_schneider",
    policy_specs: Optional[Sequence[PolicySpec]] = None,
    fixed_levels: Sequence[float] = DEFAULT_FIXED_LEVELS,
) -> List[RunConfig]:
    if time_budget_seconds is None or time_budget_seconds <= 0:
        raise ValueError("time_budget_seconds must be a positive number.")

    if policy_specs is None:
        policy_specs = default_policy_specs()

    normalized_specs: List[PolicySpec] = []
    for spec in policy_specs:
        if spec.name == "fixed":
            normalized_specs.append(FixedPartialPolicy(fixed_levels).spec)
        else:
            normalized_specs.append(spec)

    configs: List[RunConfig] = []
    for path in preset_instance_paths(preset, data_dir=data_dir):
        for spec in normalized_specs:
            for seed in seeds:
                configs.append(
                    RunConfig(
                        instance_path=path,
                        policy_spec=spec,
                        time_budget_seconds=time_budget_seconds,
                        remove_fraction=remove_fraction,
                        seed=seed,
                    )
                )
    return configs


def _run_id(run_result: RunResult) -> str:
    stem = Path(run_result.config.instance_path).stem
    return f"{stem}__{run_result.config.policy_spec.name}__seed{run_result.config.seed}"


def _run_row(run_result: RunResult) -> Dict[str, object]:
    solution = run_result.solution
    objective = solution.objective
    total_charge_time = sum(route.total_charge_time for route in solution.route_results)
    max_route_completion_time = max((route.completion_time for route in solution.route_results), default=0.0)
    row = {
        "run_id": _run_id(run_result),
        "instance": Path(run_result.config.instance_path).name,
        "policy": run_result.config.policy_spec.name,
        "policy_label": run_result.config.policy_spec.label,
        "seed": run_result.config.seed,
        "time_budget_seconds": run_result.config.time_budget_seconds,
        "iterations_completed": run_result.iterations_completed,
        "best_iteration": run_result.best_iteration,
        "stop_reason": run_result.stop_reason,
        "remove_fraction": run_result.config.remove_fraction,
        "status": run_result.status,
        "feasible": solution.feasible,
        "total_distance": objective.total_distance,
        "vehicle_count": objective.vehicle_count,
        "charger_visits": objective.charger_visits,
        "total_charge_time": total_charge_time,
        "max_route_completion_time": max_route_completion_time,
        "elapsed_seconds": run_result.elapsed_seconds,
        "error_message": run_result.error_message,
    }
    for key, value in run_result.validation.items():
        row[f"validation_{key}"] = value
    return row


def _serializable_routes(run_result: RunResult) -> Dict[str, object]:
    return {
        "run_id": _run_id(run_result),
        "instance": Path(run_result.config.instance_path).name,
        "policy": run_result.config.policy_spec.name,
        "seed": run_result.config.seed,
        "time_budget_seconds": run_result.config.time_budget_seconds,
        "iterations_completed": run_result.iterations_completed,
        "best_iteration": run_result.best_iteration,
        "stop_reason": run_result.stop_reason,
        "objective": {
            "total_distance": run_result.solution.objective.total_distance,
            "vehicle_count": run_result.solution.objective.vehicle_count,
            "charger_visits": run_result.solution.objective.charger_visits,
        },
        "routes": [
            {
                "customers": route_result.customers,
                "realized_nodes": route_result.realized_nodes,
                "total_distance": route_result.total_distance,
                "charger_visits": route_result.charger_visits,
                "total_charge_time": route_result.total_charge_time,
                "completion_time": route_result.completion_time,
                "events": [
                    {
                        "node_id": event.node_id,
                        "node_type": event.node_type,
                        "arrival_time": event.arrival_time,
                        "service_start_time": event.service_start_time,
                        "departure_time": event.departure_time,
                        "soc_arrival": event.soc_arrival,
                        "soc_departure": event.soc_departure,
                        "charge_amount": event.charge_amount,
                    }
                    for event in route_result.events
                ],
            }
            for route_result in run_result.solution.route_results
        ],
    }


def _write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _flush_batch_outputs(
    output_dir: Path,
    run_results: Sequence[RunResult],
    write_plots: bool = False,
    plots_dir: Optional[Path] = None,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    run_rows = [_run_row(run_result) for run_result in run_results]
    routes_payload = [_serializable_routes(run_result) for run_result in run_results]
    summary_by_instance_policy = summarize_by_instance_policy(run_rows)
    summary_overall = summarize_overall(summary_by_instance_policy)

    _write_csv(output_dir / "runs.csv", run_rows)
    _write_csv(output_dir / "summary_by_instance_policy.csv", summary_by_instance_policy)
    _write_csv(output_dir / "summary_overall.csv", summary_overall)
    write_table_summary_tex(output_dir / "table_summary.tex", summary_by_instance_policy)
    (output_dir / "routes.json").write_text(json.dumps(routes_payload, indent=2), encoding="utf-8")

    if write_plots and plots_dir is not None:
        save_summary_plots(summary_by_instance_policy, summary_overall, plots_dir)

    return summary_by_instance_policy, summary_overall


def _best_distance(rows: Sequence[Dict[str, object]]) -> float:
    distances = [row["total_distance"] for row in rows if isinstance(row["total_distance"], (int, float))]
    return min(distances) if distances else float("inf")


def _mean(values: Sequence[float]) -> float:
    return statistics.mean(values) if values else float("nan")


def summarize_by_instance_policy(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[tuple, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(row["instance"], row["policy"])].append(row)

    summary_rows: List[Dict[str, object]] = []
    for (instance_name, policy_name), bucket in sorted(grouped.items()):
        feasible_bucket = [row for row in bucket if row["feasible"]]
        runtimes = [float(row["elapsed_seconds"]) for row in bucket]
        distances = [float(row["total_distance"]) for row in feasible_bucket]
        vehicles = [int(row["vehicle_count"]) for row in feasible_bucket]
        chargers = [int(row["charger_visits"]) for row in feasible_bucket]
        charge_times = [float(row["total_charge_time"]) for row in feasible_bucket]
        completion_times = [float(row["max_route_completion_time"]) for row in feasible_bucket]

        summary_rows.append(
            {
                "instance": instance_name,
                "policy": policy_name,
                "runs": len(bucket),
                "feasible_runs": len(feasible_bucket),
                "best_distance": min(distances) if distances else float("inf"),
                "mean_distance": _mean(distances) if distances else float("inf"),
                "mean_runtime": _mean(runtimes),
                "mean_charge_time": _mean(charge_times) if charge_times else float("inf"),
                "mean_makespan": _mean(completion_times) if completion_times else float("inf"),
                "best_vehicle_count": min(vehicles) if vehicles else math.nan,
                "best_charger_visits": min(chargers) if chargers else math.nan,
            }
        )
    return summary_rows


def summarize_overall(summary_rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in summary_rows:
        grouped[row["policy"]].append(row)

    overall_rows: List[Dict[str, object]] = []
    for policy_name, bucket in sorted(grouped.items()):
        best_distances = [float(row["best_distance"]) for row in bucket if math.isfinite(float(row["best_distance"]))]
        mean_runtimes = [float(row["mean_runtime"]) for row in bucket]
        mean_charge_times = [float(row["mean_charge_time"]) for row in bucket if math.isfinite(float(row["mean_charge_time"]))]
        feasible_instances = sum(1 for row in bucket if row["feasible_runs"] > 0)

        overall_rows.append(
            {
                "policy": policy_name,
                "instances": len(bucket),
                "instances_with_feasible_solution": feasible_instances,
                "mean_best_distance": _mean(best_distances) if best_distances else float("inf"),
                "mean_runtime": _mean(mean_runtimes),
                "mean_charge_time": _mean(mean_charge_times) if mean_charge_times else float("inf"),
            }
        )
    return overall_rows


def write_table_summary_tex(path: Path, summary_rows: Sequence[Dict[str, object]]) -> None:
    lines = [
        "\\begin{tabular}{l l r r r r}",
        "\\hline",
        "Instance & Policy & Runs & Feasible & Best dist. & Mean runtime \\\\",
        "\\hline",
    ]
    for row in summary_rows:
        best_distance = "--" if not math.isfinite(float(row["best_distance"])) else f"{float(row['best_distance']):.2f}"
        lines.append(
            f"{row['instance']} & {row['policy']} & {row['runs']} & {row['feasible_runs']} & "
            f"{best_distance} & {float(row['mean_runtime']):.2f} \\\\"
        )
    lines.extend(["\\hline", "\\end{tabular}"])
    path.write_text("\n".join(lines), encoding="utf-8")


def save_summary_plots(
    summary_by_instance_policy: Sequence[Dict[str, object]],
    summary_overall: Sequence[Dict[str, object]],
    output_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    if summary_overall:
        policies = [row["policy"] for row in summary_overall]
        mean_best_distances = [float(row["mean_best_distance"]) for row in summary_overall]
        mean_runtimes = [float(row["mean_runtime"]) for row in summary_overall]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(policies, mean_best_distances, color=["#3465a4", "#4e9a06", "#cc0000"][: len(policies)])
        ax.set_ylabel("Mean best distance")
        ax.set_title("Policy comparison: mean best distance")
        plt.tight_layout()
        fig.savefig(output_dir / "policy_mean_best_distance.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(policies, mean_runtimes, color=["#75507b", "#f57900", "#73d216"][: len(policies)])
        ax.set_ylabel("Mean runtime (s)")
        ax.set_title("Policy comparison: mean runtime")
        plt.tight_layout()
        fig.savefig(output_dir / "policy_mean_runtime.png", dpi=180)
        plt.close(fig)

    if summary_by_instance_policy:
        fig, ax = plt.subplots(figsize=(10, 6))
        labels = [f"{row['instance']}:{row['policy']}" for row in summary_by_instance_policy[:18]]
        values = [float(row["best_distance"]) for row in summary_by_instance_policy[:18]]
        ax.barh(labels, values, color="#204a87")
        ax.set_xlabel("Best distance")
        ax.set_title("Best distance by instance/policy")
        plt.tight_layout()
        fig.savefig(output_dir / "instance_policy_best_distance.png", dpi=180)
        plt.close(fig)


def save_route_plot(run_result: RunResult, output_path: Path) -> None:
    if not run_result.solution.feasible:
        return

    instance = read_schneider_instance(run_result.config.instance_path)
    figure = plot_solution(
        instance,
        run_result.solution,
        title=f"{Path(run_result.config.instance_path).stem} - {run_result.config.policy_spec.label}",
    )
    figure.savefig(output_path, dpi=180)
    import matplotlib.pyplot as plt

    plt.close(figure)


def validate_with_exact_model(instance_path: str, time_limit: int = 60) -> Dict[str, object]:
    try:
        from pyomo.environ import SolverFactory
        from pyomo.opt import SolverStatus, TerminationCondition

        import main_3 as exact_main
    except Exception as exc:
        return {"status": "unavailable", "message": str(exc)}

    try:
        instance = read_schneider_instance(instance_path)
        if len(instance.customers) > 15:
            return {
                "source": "main_3.py:create_model_pruned",
                "status": "skipped",
                "message": "Exact validation is only configured for small Schneider instances (<= 15 customers).",
            }

        model, node_ids, depot_idx, end_idx, _, _, distance_matrix, num_vehicles = exact_main.create_model_pruned(
            instance_path,
            num_veh=len(instance.customers),
            objective_mode="distance",
        )
        solver = SolverFactory("gurobi")
        solver.options.update({"TimeLimit": time_limit, "MIPGap": 0.15})
        results = solver.solve(model, tee=False)
        status = str(results.solver.status)
        termination = str(results.solver.termination_condition)
        feasible = results.solver.status == SolverStatus.ok and results.solver.termination_condition in {
            TerminationCondition.optimal,
            TerminationCondition.maxTimeLimit,
            TerminationCondition.feasible,
        }
        travel_distance = math.nan
        used_vehicles = math.nan
        model_objective = math.nan
        if feasible:
            routes = exact_main.extract_routes(model, node_ids, depot_idx, end_idx, num_vehicles)
            node_index = {node_id: idx for idx, node_id in enumerate(node_ids)}
            travel_distance = sum(
                float(distance_matrix[node_index[source_id], node_index[target_id]])
                for arcs in routes.values()
                for source_id, target_id in arcs
            )
            used_vehicles = sum(1 for arcs in routes.values() if arcs)
            model_objective = float(model.obj())
        return {
            "source": "main_3.py:create_model_pruned",
            "status": status,
            "termination": termination,
            "feasible": feasible,
            "model_objective": model_objective,
            "travel_distance": travel_distance,
            "used_vehicles": used_vehicles,
        }
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


def run_batch(
    configs: Sequence[RunConfig],
    preset_name: str,
    output_root: str = "results",
    include_validation: bool = False,
    progress_callback: Optional[Callable[[Dict[str, object]], None]] = None,
) -> BatchSummary:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_root) / f"{timestamp}_{preset_name}"
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    manifest = _batch_manifest(preset_name, output_dir, include_validation, configs)
    (output_dir / "batch_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    run_results: List[RunResult] = []

    for index, config in enumerate(configs, start=1):
        if progress_callback:
            progress_callback(
                {
                    "type": "log",
                    "message": (
                        f"[{index}/{len(configs)}] Running {Path(config.instance_path).name} "
                        f"with {config.policy_spec.label} (seed={config.seed})"
                    ),
                }
            )

        def nested_progress(payload: Dict[str, object]) -> None:
            if progress_callback:
                forwarded = dict(payload)
                forwarded["type"] = "solver_progress"
                forwarded["instance"] = Path(config.instance_path).name
                forwarded["policy"] = config.policy_spec.name
                forwarded["seed"] = config.seed
                progress_callback(forwarded)

        run_result = solve_run(config, progress_callback=nested_progress)
        if include_validation and config.policy_spec.name == "full":
            run_result.validation = validate_with_exact_model(config.instance_path)
        run_results.append(run_result)

        if run_result.solution.feasible:
            save_route_plot(run_result, plots_dir / f"{_run_id(run_result)}_route.png")

        summary_by_instance_policy, summary_overall = _flush_batch_outputs(output_dir, run_results)

        if progress_callback:
            progress_callback(
                {
                    "type": "run_complete",
                    "run_result": run_result,
                    "message": (
                        f"Completed {Path(config.instance_path).name} / {config.policy_spec.name}: "
                        f"{objective_string(run_result.solution.objective)} "
                        f"[iters={run_result.iterations_completed}, stop={run_result.stop_reason}]"
                    ),
                }
            )

    summary_by_instance_policy, summary_overall = _flush_batch_outputs(
        output_dir,
        run_results,
        write_plots=True,
        plots_dir=plots_dir,
    )

    return BatchSummary(
        preset_name=preset_name,
        output_dir=str(output_dir),
        runs=run_results,
        summary_by_instance_policy=summary_by_instance_policy,
        summary_overall=summary_overall,
    )


def _policy_specs_from_names(policy_names: Sequence[str], fixed_levels: Sequence[float]) -> List[PolicySpec]:
    specs: List[PolicySpec] = []
    for name in policy_names:
        if name == "full":
            specs.append(FullChargePolicy().spec)
        elif name == "fixed":
            specs.append(FixedPartialPolicy(fixed_levels).spec)
        elif name == "continuous":
            specs.append(ContinuousPartialPolicy().spec)
        else:
            raise ValueError(f"Unknown policy name: {name}")
    return specs


def _main() -> None:
    parser = argparse.ArgumentParser(description="Batch experiment runner for Schneider EVRPTW instances.")
    parser.add_argument("--preset", choices=["sanity", "pilot", "paper_subset", "paper_full", "validation"], default="pilot")
    parser.add_argument("--time-budget-seconds", type=float, default=600.0)
    parser.add_argument("--remove-fraction", type=float, default=0.2)
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--policies", default="full,fixed,continuous")
    parser.add_argument("--fixed-levels", default="0.5,0.8,1.0")
    parser.add_argument("--output-root", default="results")
    parser.add_argument("--include-validation", action="store_true")
    args = parser.parse_args()

    seeds = [int(token.strip()) for token in args.seeds.split(",") if token.strip()]
    fixed_levels = [float(token.strip()) for token in args.fixed_levels.split(",") if token.strip()]
    policy_names = [token.strip() for token in args.policies.split(",") if token.strip()]
    policy_specs = _policy_specs_from_names(policy_names, fixed_levels)
    configs = build_run_configs(
        preset=args.preset,
        time_budget_seconds=args.time_budget_seconds,
        remove_fraction=args.remove_fraction,
        seeds=seeds,
        policy_specs=policy_specs,
        fixed_levels=fixed_levels,
    )

    def cli_progress(payload: Dict[str, object]) -> None:
        payload_type = payload.get("type")
        if payload_type in {"log", "run_complete"}:
            message = payload.get("message")
            if message:
                print(message)
            return
        if payload_type == "solver_progress":
            event = payload.get("event")
            if event == "new_best":
                print(
                    f"{payload.get('instance')} / {payload.get('policy')} "
                    f"iter {payload.get('iteration')}: {payload.get('objective')}"
                )
            elif event == "iteration" and int(payload.get("iteration", 0)) % 5000 == 0:
                print(
                    f"{payload.get('instance')} / {payload.get('policy')} "
                    f"iter {payload.get('iteration')}: current={payload.get('objective')} "
                    f"best={payload.get('best_objective')}"
                )

    summary = run_batch(
        configs,
        preset_name=args.preset,
        output_root=args.output_root,
        include_validation=args.include_validation,
        progress_callback=cli_progress,
    )
    print(f"Saved results to {summary.output_dir}")


if __name__ == "__main__":
    _main()
