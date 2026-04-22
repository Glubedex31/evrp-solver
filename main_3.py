import argparse
import csv
import glob
import json
import os
import time
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
from pyomo.environ import (
    Binary,
    ConcreteModel,
    Constraint,
    NonNegativeReals,
    Objective,
    RangeSet,
    Set,
    SolverFactory,
    Var,
    minimize,
)
from pyomo.opt import TerminationCondition

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_schneider_instance(filename):

    nodes = {}
    depot = None

    with open(filename, "r", encoding="utf-8") as handle:
        lines = handle.readlines()

    reading_nodes = True

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.startswith("Q"):
            reading_nodes = False

        if not reading_nodes:
            continue

        parts = line.split()
        if len(parts) < 8:
            continue
        if parts[0] == "StringID":
            continue
        if parts[1] not in ["d", "f", "c"]:
            continue

        nid = parts[0]
        raw_type = parts[1]
        x = float(parts[2])
        y = float(parts[3])
        demand = float(parts[4])
        ready = float(parts[5])
        due = float(parts[6])
        service = float(parts[7])

        if raw_type == "d":
            ntype = "DEPOT"
            depot = nid
        elif raw_type == "f":
            ntype = "CHARGER"
        else:
            ntype = "CUSTOMER"

        nodes[nid] = {
            "x": x,
            "y": y,
            "demand": demand,
            "ready": ready,
            "due": due,
            "service": service,
            "type": ntype,
        }

    battery_capacity = None
    cargo_capacity = None
    inv_ref_rate = None
    energy_rate = 1.0

    for line in lines:
        line = line.strip()
        if "/" not in line:
            continue

        try:
            value = float(line.split("/")[1])
        except Exception:
            continue

        first_word = line.split()[0]
        if first_word == "Q":
            battery_capacity = value
        elif first_word == "C":
            cargo_capacity = value
        elif first_word == "r":
            energy_rate = value
        elif first_word == "g":
            inv_ref_rate = value

    if depot is None or battery_capacity is None or cargo_capacity is None:
        raise ValueError("Failed to read Schneider instance.")

    return {
        "nodes": nodes,
        "depot": depot,
        "cargo_capacity": cargo_capacity,
        "battery_capacity": battery_capacity,
        "energy_rate": energy_rate,
        "inv_ref_rate": inv_ref_rate if inv_ref_rate is not None else 1.0,
    }


def compute_distance(nodes):
    node_ids = list(nodes.keys())
    coords = np.array([[node["x"], node["y"]] for node in nodes.values()])
    diff = coords[:, None, :] - coords[None, :, :]
    dist_matrix = np.sqrt(np.sum(diff * diff, axis=2))
    return dist_matrix, node_ids


def create_model_pruned(file, num_veh=1, objective_mode="legacy_cost"):
    data = read_schneider_instance(file)
    nodes = dict(data["nodes"])
    depot_id = data["depot"]
    if depot_id is None:
        raise ValueError("No depot found in Schneider instance.")

    cargo_capacity = data["cargo_capacity"]
    battery_capacity = data["battery_capacity"]
    energy_rate = data["energy_rate"]
    inv_ref_rate = data["inv_ref_rate"]

    end_depot = depot_id + "_end"
    if end_depot in nodes:
        raise ValueError(f"Node {end_depot} already exists.")
    nodes[end_depot] = {
        **nodes[depot_id],
        "demand": 0.0,
        "service": 0.0,
        "type": "DEPOT",
    }

    dist_matrix, node_ids = compute_distance(nodes)

    n = len(node_ids)
    all_nodes = list(range(n))
    depot_idx = node_ids.index(depot_id)
    end_idx = node_ids.index(end_depot)

    demand = {i: nodes[node_ids[i]]["demand"] for i in all_nodes}
    customers = [i for i in all_nodes if demand[i] > 0]
    stations = [i for i in all_nodes if nodes[node_ids[i]]["type"] == "CHARGER"]

    ready = {i: nodes[node_ids[i]]["ready"] for i in all_nodes}
    due = {i: nodes[node_ids[i]]["due"] for i in all_nodes}
    service = {i: nodes[node_ids[i]]["service"] for i in all_nodes}

    max_dist = float(np.max(dist_matrix))
    max_consumption = energy_rate * max_dist
    big_m_energy = battery_capacity + max_consumption
    big_m_time = max(due.values()) + max_dist + max(service.values())
    big_m = max(big_m_energy, big_m_time)

    feasible_arcs = []
    for i in all_nodes:
        for j in all_nodes:
            if i == j:
                continue
            if j == depot_idx or i == end_idx:
                continue
            dij = dist_matrix[i, j]
            if energy_rate * dij > battery_capacity:
                continue
            if ready[i] + service[i] + dij <= due[j]:
                feasible_arcs.append((i, j))

    if num_veh is None:
        num_veh = len(customers)

    model = ConcreteModel()
    model.I = Set(initialize=all_nodes)
    model.C = Set(initialize=customers)
    model.K = RangeSet(0, num_veh - 1)
    model.A = Set(initialize=feasible_arcs, dimen=2)
    model.S = Set(initialize=stations)

    model.delta = Var(model.A, model.K, domain=Binary)
    model.use_vehicle = Var(model.K, domain=Binary)
    model.e = Var(model.I, model.K, domain=NonNegativeReals, bounds=(0, battery_capacity))
    model.t = Var(model.I, model.K, domain=NonNegativeReals)
    model.u = Var(model.I, model.K, domain=NonNegativeReals, bounds=(0, n - 1))

    model.B = battery_capacity
    model.energy_rate = energy_rate
    model.g = inv_ref_rate
    model.dist_matrix = dist_matrix

    def obj_rule(m):
        travel_distance = sum(dist_matrix[i, j] * m.delta[i, j, k] for (i, j) in m.A for k in m.K)

        if objective_mode == "distance":
            return travel_distance

        if objective_mode == "legacy_cost":
            price = 0.2
            vehicle_cost = 150.0
            energy_cost = sum(price * energy_rate * dist_matrix[i, j] * m.delta[i, j, k] for (i, j) in m.A for k in m.K)
            fleet_cost = sum(vehicle_cost * m.use_vehicle[k] for k in m.K)
            return travel_distance + energy_cost + fleet_cost

        raise ValueError(f"Unsupported objective_mode: {objective_mode}")

    model.obj = Objective(rule=obj_rule, sense=minimize)

    def visit_rule(m, i):
        return sum(m.delta[i, j, k] for (ii, j) in m.A if ii == i for k in m.K) == 1

    model.visit = Constraint(model.C, rule=visit_rule)

    def depart_rule(m, k):
        return sum(m.delta[depot_idx, j, k] for (i, j) in m.A if i == depot_idx) == m.use_vehicle[k]

    model.depart = Constraint(model.K, rule=depart_rule)

    def arrive_rule(m, k):
        return sum(m.delta[i, end_idx, k] for (i, j) in m.A if j == end_idx) == m.use_vehicle[k]

    model.arrive = Constraint(model.K, rule=arrive_rule)

    def flow_rule(m, j, k):
        if j in [depot_idx, end_idx]:
            return Constraint.Skip
        return (
            sum(m.delta[i, j, k] for (i, jj) in m.A if jj == j)
            - sum(m.delta[j, h, k] for (ii, h) in m.A if ii == j)
        ) == 0

    model.flow = Constraint(model.I, model.K, rule=flow_rule)

    def cap_rule(m, k):
        return sum(demand[i] * m.delta[i, j, k] for (i, j) in m.A) <= cargo_capacity

    model.capacity = Constraint(model.K, rule=cap_rule)

    def time_window_rule(m, i, k):
        return ready[i], m.t[i, k], due[i]

    model.time_window = Constraint(model.I, model.K, rule=time_window_rule)

    def depot_time_rule(m, k):
        return m.t[depot_idx, k] == 0

    model.depot_time = Constraint(model.K, rule=depot_time_rule)

    def time_arc_rule(m, i, j, k):
        travel = dist_matrix[i, j]
        if j in m.S:
            recharge_time = inv_ref_rate * (battery_capacity - (m.e[i, k] - energy_rate * travel))
        else:
            recharge_time = 0
        return m.t[j, k] >= m.t[i, k] + service[i] + travel + recharge_time - big_m * (1 - m.delta[i, j, k])

    model.time_arc = Constraint(model.A, model.K, rule=time_arc_rule)

    def battery_cap_rule(m, i, k):
        return m.e[i, k] <= battery_capacity

    model.bat_cap = Constraint(model.I, model.K, rule=battery_cap_rule)

    def enough_energy_rule(m, i, j, k):
        return m.e[i, k] >= energy_rate * dist_matrix[i, j] - big_m * (1 - m.delta[i, j, k])

    model.energy_depart = Constraint(model.A, model.K, rule=enough_energy_rule)

    def battery_transition_upper(m, i, j, k):
        if j in m.S:
            return Constraint.Skip
        consumption = energy_rate * dist_matrix[i, j]
        return m.e[j, k] <= m.e[i, k] - consumption + big_m * (1 - m.delta[i, j, k])

    model.battery_transition_upper = Constraint(model.A, model.K, rule=battery_transition_upper)

    def battery_transition_lower(m, i, j, k):
        if j in m.S:
            return Constraint.Skip
        consumption = energy_rate * dist_matrix[i, j]
        return m.e[j, k] >= m.e[i, k] - consumption - big_m * (1 - m.delta[i, j, k])

    model.battery_transition_lower = Constraint(model.A, model.K, rule=battery_transition_lower)

    def init_battery_rule(m, k):
        return m.e[depot_idx, k] == battery_capacity

    model.init_battery = Constraint(model.K, rule=init_battery_rule)

    def full_recharge_rule(m, i, j, k):
        if j not in m.S:
            return Constraint.Skip
        return m.e[j, k] >= battery_capacity - big_m * (1 - m.delta[i, j, k])

    model.full_recharge = Constraint(model.A, model.K, rule=full_recharge_rule)

    def mtz_rule(m, i, j, k):
        if i == depot_idx or j == depot_idx:
            return Constraint.Skip
        return m.u[i, k] + 1 - n * (1 - m.delta[i, j, k]) <= m.u[j, k]

    model.mtz = Constraint(model.A, model.K, rule=mtz_rule)

    def depot_order_rule(m, k):
        return m.u[depot_idx, k] == 0

    model.depot_order = Constraint(model.K, rule=depot_order_rule)

    def symmetry_break_rule(m, k):
        if k == 0:
            return Constraint.Skip
        return m.use_vehicle[k] <= m.use_vehicle[k - 1]

    model.symmetry = Constraint(model.K, rule=symmetry_break_rule)

    return model, node_ids, depot_idx, end_idx, stations, nodes, dist_matrix, num_veh


def extract_routes(model, node_ids, depot_idx, end_idx, num_vehicles):
    routes = {k: [] for k in range(num_vehicles)}
    successors = {k: {} for k in range(num_vehicles)}

    for (i, j) in model.A:
        for k in model.K:
            value = model.delta[i, j, k].value
            if value is not None and value > 0.5:
                successors[k][i] = j

    for k in range(num_vehicles):
        if depot_idx not in successors[k]:
            continue

        current = depot_idx
        visited = set()
        while current != end_idx:
            if current in visited:
                print("WARNING: subtour detected")
                break
            visited.add(current)

            if current not in successors[k]:
                break

            nxt = successors[k][current]
            routes[k].append((node_ids[current], node_ids[nxt]))
            current = nxt

    return routes


def print_routes_with_battery(model, routes, node_ids):
    energy_rate = model.energy_rate
    dist_matrix = model.dist_matrix

    for veh, arcs in routes.items():
        print(f"Vehicle {veh + 1} route:")
        if not arcs:
            print("  Idle")
            continue

        for source_node, target_node in arcs:
            i = node_ids.index(source_node)
            j = node_ids.index(target_node)
            depart = model.e[i, veh].value or 0.0
            consumption = energy_rate * dist_matrix[i, j]
            arrival = max(depart - consumption, 0)

            if j in model.S:
                after = model.e[j, veh].value or 0.0
                print(f"  {source_node} -> {target_node}: {depart:.1f} -> {arrival:.1f} -> {after:.1f}  [charge]")
            else:
                print(f"  {source_node} -> {target_node}: {depart:.1f} -> {arrival:.1f}")


def plot_solution(model, routes, nodes, node_ids, num_vehicles, output_path=None, show=False):
    battery_capacity = model.B
    fig, ax = plt.subplots(figsize=(11, 9))
    used_labels = set()

    for node_id in node_ids:
        x, y = nodes[node_id]["x"], nodes[node_id]["y"]
        node_type = nodes[node_id]["type"]

        if node_type == "DEPOT":
            color, marker, label = "blue", "s", "Depot"
        elif node_type == "CUSTOMER":
            color, marker, label = "gold", "o", "Customer"
        else:
            color, marker, label = "red", "^", "Charger"

        if label not in used_labels:
            ax.scatter(x, y, color=color, marker=marker, s=100, label=label)
            used_labels.add(label)
        else:
            ax.scatter(x, y, color=color, marker=marker, s=100)

    colors = plt.cm.tab20.colors

    for vehicle in range(num_vehicles):
        arcs = routes[vehicle]
        if not arcs:
            continue

        color = colors[vehicle % len(colors)]
        first = True

        for source_node, target_node in arcs:
            i = node_ids.index(source_node)
            j = node_ids.index(target_node)
            x1, y1 = nodes[source_node]["x"], nodes[source_node]["y"]
            x2, y2 = nodes[target_node]["x"], nodes[target_node]["y"]

            depart = model.e[i, vehicle].value or 0.0
            consumption = model.energy_rate * model.dist_matrix[i, j]
            arrival = max(depart - consumption, 0)

            perc_depart = 100 * depart / battery_capacity
            perc_arrive = 100 * arrival / battery_capacity
            if j in model.S:
                after = model.e[j, vehicle].value or 0.0
                perc_after = 100 * after / battery_capacity
                label = f"{perc_depart:.0f}% -> {perc_arrive:.0f}% -> {perc_after:.0f}%"
            else:
                label = f"{perc_depart:.0f}% -> {perc_arrive:.0f}%"

            if first:
                ax.plot([x1, x2], [y1, y2], "-", color=color, label=f"Vehicle {vehicle + 1}")
                first = False
            else:
                ax.plot([x1, x2], [y1, y2], "-", color=color)

            ax.text((x1 + x2) / 2, (y1 + y2) / 2, label, fontsize=8, color=color)

    ax.legend()
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("EVRP Solution with Battery Levels (%)")
    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=180)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def list_schneider_instances(data_dir="data_schneider"):
    paths = []
    for path in sorted(glob.glob(os.path.join(data_dir, "*.txt"))):
        if os.path.basename(path).lower() == "readme.txt":
            continue
        paths.append(path)
    return paths


def preset_instance_paths(preset, data_dir="data_schneider"):
    paths = list_schneider_instances(data_dir)
    names = {Path(path).name: path for path in paths}

    if preset == "sanity":
        return [path for path in paths if any(tag in Path(path).stem for tag in ("C5", "C10", "C15"))]

    if preset == "validation":
        validation_names = ["c101C5.txt", "r102C10.txt", "rc108C5.txt"]
        return [names[name] for name in validation_names if name in names]

    if preset == "paper_full":
        return paths

    raise ValueError(f"Unknown preset: {preset}")


def _safe_float(value):
    try:
        return float(value)
    except Exception:
        return float("nan")


def _has_solution_values(model):
    for (i, j) in model.A:
        for k in model.K:
            if model.delta[i, j, k].value is not None:
                return True
    return False


def build_route_details(model, routes, nodes, node_ids):
    route_details = []
    station_indices = set(model.S)

    for vehicle, arcs in routes.items():
        if not arcs:
            continue

        realized_nodes = [arcs[0][0]]
        arc_details = []
        total_distance = 0.0

        for source_id, target_id in arcs:
            source_idx = node_ids.index(source_id)
            target_idx = node_ids.index(target_id)

            depart_soc = _safe_float(model.e[source_idx, vehicle].value or 0.0)
            distance = _safe_float(model.dist_matrix[source_idx, target_idx])
            energy_consumption = _safe_float(model.energy_rate) * distance
            arrival_soc = max(depart_soc - energy_consumption, 0.0)
            target_soc = _safe_float(model.e[target_idx, vehicle].value or arrival_soc)
            charge_amount = max(target_soc - arrival_soc, 0.0) if target_idx in station_indices else 0.0

            arc_details.append(
                {
                    "from_node": source_id,
                    "to_node": target_id,
                    "to_node_type": nodes[target_id]["type"],
                    "distance": distance,
                    "energy_consumption": energy_consumption,
                    "soc_departure": depart_soc,
                    "soc_arrival_before_charge": arrival_soc,
                    "soc_target_state": target_soc,
                    "charge_amount_at_target": charge_amount,
                }
            )
            realized_nodes.append(target_id)
            total_distance += distance

        route_details.append(
            {
                "vehicle_index": int(vehicle),
                "nodes": realized_nodes,
                "arcs": arc_details,
                "total_distance": total_distance,
            }
        )

    return route_details


def solve_exact_instance(
    instance_path,
    num_veh=None,
    objective_mode="distance",
    solver_name="gurobi",
    time_limit=60,
    mip_gap=0.15,
    tee=False,
    save_plot=False,
    plot_dir=None,
):
    instance_data = read_schneider_instance(instance_path)
    customer_count = sum(1 for node in instance_data["nodes"].values() if node["type"] == "CUSTOMER")
    charger_count = sum(1 for node in instance_data["nodes"].values() if node["type"] == "CHARGER")
    vehicle_limit = customer_count if num_veh is None else num_veh

    start_time = time.time()
    model, node_ids, depot_idx, end_idx, stations, nodes, dist_matrix, num_vehicles = create_model_pruned(
        instance_path,
        num_veh=vehicle_limit,
        objective_mode=objective_mode,
    )

    solver = SolverFactory(solver_name)
    solver.options.update({"MIPGap": mip_gap, "TimeLimit": time_limit})
    results = solver.solve(model, tee=tee, load_solutions=False)
    elapsed_seconds = time.time() - start_time

    status = str(results.solver.status)
    termination = str(results.solver.termination_condition)
    solver_message = str(getattr(results.solver, "message", "") or "")
    has_incumbent = len(results.solution) > 0
    load_error = ""

    if has_incumbent:
        try:
            model.solutions.load_from(results)
        except Exception as exc:
            load_error = str(exc)

    accepted_terminations = {
        TerminationCondition.optimal,
        TerminationCondition.maxTimeLimit,
        TerminationCondition.feasible,
    }
    feasible = (
        has_incumbent
        and not load_error
        and _has_solution_values(model)
        and results.solver.termination_condition in accepted_terminations
    )

    if feasible:
        routes = extract_routes(model, node_ids, depot_idx, end_idx, num_vehicles)
        route_details = build_route_details(model, routes, nodes, node_ids)
    else:
        routes = {k: [] for k in range(num_vehicles)}
        route_details = []

    travel_distance = sum(route["total_distance"] for route in route_details)
    used_vehicles = len(route_details)
    objective_value = _safe_float(model.obj()) if feasible else float("nan")

    plot_file = ""
    if save_plot and feasible and plot_dir is not None:
        plot_path = Path(plot_dir) / f"{Path(instance_path).stem}_route.png"
        plot_solution(model, routes, nodes, node_ids, num_vehicles, output_path=plot_path, show=False)
        plot_file = str(plot_path)

    run_row = {
        "instance": Path(instance_path).name,
        "status": status,
        "termination": termination,
        "feasible": feasible,
        "objective_mode": objective_mode,
        "objective_value": objective_value,
        "travel_distance": travel_distance if feasible else float("nan"),
        "used_vehicles": used_vehicles if feasible else float("nan"),
        "vehicle_limit": num_vehicles,
        "customers": customer_count,
        "chargers": charger_count,
        "time_limit": time_limit,
        "mip_gap": mip_gap,
        "runtime_seconds": elapsed_seconds,
        "solver": solver_name,
        "plot_file": plot_file,
        "has_incumbent": has_incumbent,
        "solver_message": solver_message,
        "load_error": load_error,
    }

    route_payload = {
        "instance": Path(instance_path).name,
        "status": status,
        "termination": termination,
        "feasible": feasible,
        "has_incumbent": has_incumbent,
        "solver_message": solver_message,
        "load_error": load_error,
        "objective_value": objective_value,
        "travel_distance": travel_distance if feasible else float("nan"),
        "used_vehicles": used_vehicles if feasible else float("nan"),
        "routes": route_details,
    }
    return run_row, route_payload


def _write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_csv(path, rows):
    feasible_rows = [row for row in rows if row["feasible"]]
    optimal_rows = [row for row in rows if str(row["termination"]).lower() == "optimal"]
    time_limited_rows = [row for row in rows if str(row["termination"]).lower() == "maxtimelimit"]

    summary_row = {
        "instances": len(rows),
        "feasible_instances": len(feasible_rows),
        "optimal_instances": len(optimal_rows),
        "time_limited_instances": len(time_limited_rows),
        "mean_travel_distance": float(np.mean([row["travel_distance"] for row in feasible_rows])) if feasible_rows else float("nan"),
        "mean_runtime_seconds": float(np.mean([row["runtime_seconds"] for row in rows])) if rows else float("nan"),
    }
    _write_csv(path, [summary_row])


def write_report(path, rows, output_dir):
    lines = [
        "main_3 exact-validator report",
        f"Output directory: {output_dir}",
        "",
        "Interpretation:",
        "- feasible=True means the MILP produced a route set that can be extracted.",
        "- termination=optimal means proven optimal for the chosen model and limits.",
        "- termination=maxTimeLimit with feasible=True means a feasible incumbent was found but not proven optimal.",
        "- travel_distance is the distance of the extracted routes.",
        "- objective_value equals travel_distance when objective_mode=distance.",
        "",
        "Per-instance summary:",
    ]

    for row in rows:
        distance_text = "n/a"
        if row["feasible"]:
            distance_text = f"{float(row['travel_distance']):.2f}"
        lines.append(
            f"- {row['instance']}: feasible={row['feasible']}, status={row['status']}, "
            f"termination={row['termination']}, incumbent={row.get('has_incumbent', False)}, "
            f"distance={distance_text}, runtime={float(row['runtime_seconds']):.2f}s"
        )
        if row.get("solver_message"):
            lines.append(f"  solver_message: {row['solver_message']}")
        if row.get("load_error"):
            lines.append(f"  load_error: {row['load_error']}")

    path.write_text("\n".join(lines), encoding="utf-8")


def run_validation_batch(
    instance_paths,
    output_root="results",
    label="main3_validation",
    num_veh=None,
    objective_mode="distance",
    solver_name="gurobi",
    time_limit=60,
    mip_gap=0.15,
    tee=False,
    save_plots=True,
):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_root) / f"{timestamp}_{label}"
    plots_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    run_rows = []
    route_payloads = []

    for index, instance_path in enumerate(instance_paths, start=1):
        print(f"[{index}/{len(instance_paths)}] Solving {Path(instance_path).name}")
        try:
            run_row, route_payload = solve_exact_instance(
                instance_path,
                num_veh=num_veh,
                objective_mode=objective_mode,
                solver_name=solver_name,
                time_limit=time_limit,
                mip_gap=mip_gap,
                tee=tee,
                save_plot=save_plots,
                plot_dir=plots_dir,
            )
        except Exception as exc:
            run_row = {
                "instance": Path(instance_path).name,
                "status": "error",
                "termination": "error",
                "feasible": False,
                "objective_mode": objective_mode,
                "objective_value": float("nan"),
                "travel_distance": float("nan"),
                "used_vehicles": float("nan"),
                "vehicle_limit": num_veh if num_veh is not None else float("nan"),
                "customers": float("nan"),
                "chargers": float("nan"),
                "time_limit": time_limit,
                "mip_gap": mip_gap,
                "runtime_seconds": 0.0,
                "solver": solver_name,
                "plot_file": "",
                "has_incumbent": False,
                "solver_message": "",
                "load_error": str(exc),
            }
            route_payload = {
                "instance": Path(instance_path).name,
                "status": "error",
                "termination": "error",
                "feasible": False,
                "has_incumbent": False,
                "solver_message": "",
                "load_error": str(exc),
                "objective_value": float("nan"),
                "travel_distance": float("nan"),
                "used_vehicles": float("nan"),
                "routes": [],
            }
        run_rows.append(run_row)
        route_payloads.append(route_payload)
        print(
            f"  status={run_row['status']}, termination={run_row['termination']}, "
            f"feasible={run_row['feasible']}, runtime={float(run_row['runtime_seconds']):.2f}s"
        )
        if run_row.get("solver_message"):
            print(f"  solver_message={run_row['solver_message']}")
        if run_row.get("load_error"):
            print(f"  load_error={run_row['load_error']}")

    _write_csv(output_dir / "runs.csv", run_rows)
    (output_dir / "routes.json").write_text(json.dumps(route_payloads, indent=2), encoding="utf-8")
    write_summary_csv(output_dir / "summary.csv", run_rows)
    write_report(output_dir / "report.txt", run_rows, output_dir)
    return output_dir


def resolve_instance_path(instance_arg, data_dir="data_schneider"):
    candidate = Path(instance_arg)
    if candidate.exists():
        return str(candidate)

    data_candidate = Path(data_dir) / instance_arg
    if data_candidate.exists():
        return str(data_candidate)

    raise FileNotFoundError(f"Could not find instance: {instance_arg}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run the main_3 full-charge MILP validator on Schneider instances.")
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument("--instance", help="Single Schneider instance path or filename.")
    target_group.add_argument("--preset", choices=["sanity", "validation", "paper_full"], help="Named instance preset.")
    parser.add_argument("--data-dir", default="data_schneider", help="Directory containing Schneider instance files.")
    parser.add_argument("--output-root", default="results", help="Directory where validator outputs will be stored.")
    parser.add_argument("--num-veh", type=int, default=None, help="Vehicle limit. Default is number of customers.")
    parser.add_argument("--objective-mode", choices=["distance", "legacy_cost"], default="distance")
    parser.add_argument("--solver", default="gurobi", help="Pyomo solver name.")
    parser.add_argument("--time-limit", type=int, default=60, help="Per-instance solver time limit in seconds.")
    parser.add_argument("--mip-gap", type=float, default=0.15, help="Relative MIP gap.")
    parser.add_argument("--tee", action="store_true", help="Stream solver output.")
    parser.add_argument("--no-plots", action="store_true", help="Do not save route plot PNGs.")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.instance:
        instance_paths = [resolve_instance_path(args.instance, data_dir=args.data_dir)]
        label = f"main3_{Path(instance_paths[0]).stem}"
    else:
        instance_paths = preset_instance_paths(args.preset, data_dir=args.data_dir)
        label = f"main3_{args.preset}"

    output_dir = run_validation_batch(
        instance_paths,
        output_root=args.output_root,
        label=label,
        num_veh=args.num_veh,
        objective_mode=args.objective_mode,
        solver_name=args.solver,
        time_limit=args.time_limit,
        mip_gap=args.mip_gap,
        tee=args.tee,
        save_plots=not args.no_plots,
    )
    print(f"Saved validator outputs to {output_dir}")


if __name__ == "__main__":
    main()
