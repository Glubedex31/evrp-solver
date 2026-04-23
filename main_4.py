import argparse
import math
import random
import time
from dataclasses import dataclass, field
from heapq import heappop, heappush
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


EPSILON = 1e-9
DEFAULT_FIXED_LEVELS = [0.5, 0.8, 1.0]


class TimeBudgetExceeded(RuntimeError):
    pass


def _check_deadline(deadline: Optional[float]) -> None:
    if deadline is not None and time.perf_counter() >= deadline:
        raise TimeBudgetExceeded


@dataclass(frozen=True)
class Node:
    nid: str
    ntype: str
    x: float
    y: float
    demand: float
    ready: float
    due: float
    service: float


@dataclass
class Instance:
    nodes: Dict[str, Node]
    depot: str
    battery_capacity: float
    cargo_capacity: float
    energy_rate: float
    inv_ref_rate: float
    chargers: List[str] = field(default_factory=list)
    customers: List[str] = field(default_factory=list)
    distances: Dict[Tuple[str, str], float] = field(default_factory=dict)
    charger_rankings: Dict[str, List[str]] = field(default_factory=dict)


@dataclass(frozen=True, order=True)
class ObjectiveValue:
    total_distance: float
    vehicle_count: int
    charger_visits: int

    @classmethod
    def infinity(cls) -> "ObjectiveValue":
        return cls(float("inf"), 10**9, 10**9)


@dataclass(frozen=True)
class PolicySpec:
    name: str
    label: str
    fixed_levels: Tuple[float, ...] = ()

    @property
    def key(self) -> str:
        if self.fixed_levels:
            levels = ",".join(f"{level:.2f}" for level in self.fixed_levels)
            return f"{self.name}:{levels}"
        return self.name


@dataclass
class Route:
    customers: List[str]


@dataclass(frozen=True)
class StopEvent:
    node_id: str
    node_type: str
    arrival_time: float
    service_start_time: float
    departure_time: float
    soc_arrival: float
    soc_departure: float
    charge_amount: float


@dataclass
class RouteResult:
    customers: List[str]
    realized_nodes: List[str]
    events: List[StopEvent]
    feasible: bool
    total_distance: float
    charger_visits: int
    objective: ObjectiveValue
    message: str = ""
    total_charge_time: float = 0.0
    completion_time: float = 0.0


@dataclass
class Solution:
    routes: List[Route]
    objective: ObjectiveValue = field(default_factory=ObjectiveValue.infinity)
    feasible: bool = False
    route_results: List[RouteResult] = field(default_factory=list)
    search_stats: "SearchStats" = field(default_factory=lambda: SearchStats())


@dataclass(frozen=True)
class SearchStats:
    iterations_completed: int = 0
    best_iteration: int = 0
    stop_reason: str = ""
    configured_time_budget_seconds: Optional[float] = None


@dataclass(frozen=True)
class RunConfig:
    instance_path: str
    policy_spec: PolicySpec
    time_budget_seconds: float = 600.0
    remove_fraction: float = 0.3
    seed: int = 42


@dataclass
class RunResult:
    config: RunConfig
    solution: Solution
    elapsed_seconds: float
    status: str
    error_message: str = ""
    validation: Dict[str, object] = field(default_factory=dict)
    iterations_completed: int = 0
    best_iteration: int = 0
    stop_reason: str = ""


@dataclass
class BatchSummary:
    preset_name: str
    output_dir: str
    runs: List[RunResult]
    summary_by_instance_policy: List[Dict[str, object]] = field(default_factory=list)
    summary_overall: List[Dict[str, object]] = field(default_factory=list)


@dataclass(frozen=True)
class _Label:
    total_distance: float
    time_value: float
    soc: float
    charger_visits: int
    index: int
    node_id: str
    zero_hops: int
    route_nodes: Tuple[str, ...]


class ChargingPolicy:
    def __init__(self, spec: PolicySpec):
        self.spec = spec

    def recharge_target(self, current_soc: float, required_soc: float, battery_capacity: float) -> float:
        raise NotImplementedError

    def policy_id(self) -> str:
        return self.spec.key


class FullChargePolicy(ChargingPolicy):
    def __init__(self):
        super().__init__(PolicySpec(name="full", label="Full charge"))

    def recharge_target(self, current_soc: float, required_soc: float, battery_capacity: float) -> float:
        return battery_capacity


class FixedPartialPolicy(ChargingPolicy):
    def __init__(self, levels: Sequence[float]):
        unique_levels = tuple(sorted({float(level) for level in levels}))
        super().__init__(PolicySpec(name="fixed", label="Fixed partial", fixed_levels=unique_levels))
        self.levels = unique_levels

    def recharge_target(self, current_soc: float, required_soc: float, battery_capacity: float) -> float:
        for fraction in self.levels:
            target = fraction * battery_capacity
            if target + EPSILON >= required_soc and target + EPSILON >= current_soc:
                return target
        return battery_capacity


class ContinuousPartialPolicy(ChargingPolicy):
    def __init__(self):
        super().__init__(PolicySpec(name="continuous", label="Continuous partial"))

    def recharge_target(self, current_soc: float, required_soc: float, battery_capacity: float) -> float:
        return min(battery_capacity, max(current_soc, required_soc))


def make_policy_from_spec(spec: PolicySpec) -> ChargingPolicy:
    if spec.name == "full":
        return FullChargePolicy()
    if spec.name == "fixed":
        return FixedPartialPolicy(spec.fixed_levels or DEFAULT_FIXED_LEVELS)
    if spec.name == "continuous":
        return ContinuousPartialPolicy()
    raise ValueError(f"Unsupported policy: {spec.name}")


def default_policy_specs() -> List[PolicySpec]:
    return [
        FullChargePolicy().spec,
        FixedPartialPolicy(DEFAULT_FIXED_LEVELS).spec,
        ContinuousPartialPolicy().spec,
    ]


def read_schneider_instance(filename: str) -> Instance:
    nodes: Dict[str, Node] = {}
    depot = None

    with open(filename, "r", encoding="utf-8") as handle:
        lines = handle.readlines()

    reading_nodes = True
    battery_capacity = None
    cargo_capacity = None
    inv_ref_rate = None
    energy_rate = 1.0

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue

        if line.startswith("Q"):
            reading_nodes = False

        if reading_nodes:
            parts = line.split()
            if len(parts) < 8:
                continue
            if parts[0] == "StringID":
                continue
            if parts[1] not in {"d", "f", "c"}:
                continue

            nid = parts[0]
            raw_type = parts[1]

            if raw_type == "d":
                ntype = "DEPOT"
                depot = nid
            elif raw_type == "f":
                ntype = "CHARGER"
            else:
                ntype = "CUSTOMER"

            nodes[nid] = Node(
                nid=nid,
                ntype=ntype,
                x=float(parts[2]),
                y=float(parts[3]),
                demand=float(parts[4]),
                ready=float(parts[5]),
                due=float(parts[6]),
                service=float(parts[7]),
            )
            continue

        if "/" not in line:
            continue

        try:
            value = float(line.split("/")[1])
        except ValueError:
            continue

        head = line.split()[0]
        if head == "Q":
            battery_capacity = value
        elif head == "C":
            cargo_capacity = value
        elif head == "r":
            energy_rate = value
        elif head == "g":
            inv_ref_rate = value

    if depot is None or battery_capacity is None or cargo_capacity is None:
        raise ValueError("Failed to parse instance.")

    instance = Instance(
        nodes=nodes,
        depot=depot,
        battery_capacity=battery_capacity,
        cargo_capacity=cargo_capacity,
        energy_rate=energy_rate,
        inv_ref_rate=inv_ref_rate if inv_ref_rate is not None else 1.0,
    )
    instance.chargers = [nid for nid, node in nodes.items() if node.ntype == "CHARGER"]
    instance.customers = [nid for nid, node in nodes.items() if node.ntype == "CUSTOMER"]
    compute_distances(instance)
    return instance


def list_schneider_instances(data_dir: str = "data_schneider") -> List[str]:
    directory = Path(data_dir)
    return sorted(
        str(path)
        for path in directory.glob("*.txt")
        if path.name.lower() != "readme.txt"
    )


def compute_distances(instance: Instance) -> None:
    for source_id, source in instance.nodes.items():
        for target_id, target in instance.nodes.items():
            if source_id == target_id:
                continue
            instance.distances[(source_id, target_id)] = math.hypot(source.x - target.x, source.y - target.y)
    instance.charger_rankings = {
        node_id: sorted(
            instance.chargers,
            key=lambda charger_id: instance.distances[(node_id, charger_id)] if node_id != charger_id else 0.0,
        )
        for node_id in instance.nodes
    }


def energy_needed(instance: Instance, source_id: str, target_id: str) -> float:
    if source_id == target_id:
        return 0.0
    return instance.energy_rate * instance.distances[(source_id, target_id)]


def objective_string(objective: ObjectiveValue) -> str:
    if math.isinf(objective.total_distance):
        return "infeasible"
    return (
        f"distance={objective.total_distance:.2f}, "
        f"vehicles={objective.vehicle_count}, "
        f"chargers={objective.charger_visits}"
    )


def better_objective(candidate: ObjectiveValue, current: ObjectiveValue) -> bool:
    return candidate < current


def copy_solution(solution: Solution) -> Solution:
    return Solution(
        routes=[Route(list(route.customers)) for route in solution.routes],
        objective=solution.objective,
        feasible=solution.feasible,
        route_results=list(solution.route_results),
        search_stats=solution.search_stats,
    )


def _scheduled_temperature(initial_temperature: float, final_temperature: float, progress: float) -> float:
    progress = min(1.0, max(0.0, progress))
    if initial_temperature <= 0.0:
        return max(final_temperature, 1e-6)
    if final_temperature <= 0.0:
        final_temperature = 1e-6
    ratio = final_temperature / initial_temperature
    return max(final_temperature, initial_temperature * (ratio**progress))


def _dominates(left: _Label, right: _Label) -> bool:
    left_no_worse = (
        left.total_distance <= right.total_distance + EPSILON
        and left.time_value <= right.time_value + EPSILON
        and left.soc + EPSILON >= right.soc
        and left.charger_visits <= right.charger_visits
    )
    left_strictly_better = (
        left.total_distance < right.total_distance - EPSILON
        or left.time_value < right.time_value - EPSILON
        or left.soc > right.soc + EPSILON
        or left.charger_visits < right.charger_visits
    )
    return left_no_worse and left_strictly_better


def _customer_prefix_distances(instance: Instance, customers: Sequence[str]) -> List[float]:
    prefix = [0.0] * len(customers)
    for idx in range(1, len(customers)):
        prefix[idx] = prefix[idx - 1] + instance.distances[(customers[idx - 1], customers[idx])]
    return prefix


def _segment_distance(
    instance: Instance,
    prefix: Sequence[float],
    customers: Sequence[str],
    start_node: str,
    start_index: int,
    end_index: int,
    end_node: str,
) -> float:
    if end_index < start_index:
        return instance.distances[(start_node, end_node)]

    distance = instance.distances[(start_node, customers[start_index])]
    if end_index > start_index:
        distance += prefix[end_index] - prefix[start_index]
    distance += instance.distances[(customers[end_index], end_node)]
    return distance


def _required_energy_to_next_recharge(instance: Instance, realized_nodes: Sequence[str], start_index: int) -> float:
    required = 0.0
    for edge_index in range(start_index, len(realized_nodes) - 1):
        source = realized_nodes[edge_index]
        target = realized_nodes[edge_index + 1]
        required += energy_needed(instance, source, target)
        target_type = instance.nodes[target].ntype
        if target_type == "CHARGER" or edge_index + 1 == len(realized_nodes) - 1:
            return required
    return required


def _charger_candidates(instance: Instance, anchor_nodes: Sequence[str], limit: int = 5) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for anchor in anchor_nodes:
        for charger_id in instance.charger_rankings.get(anchor, []):
            if charger_id in seen:
                continue
            seen.add(charger_id)
            ordered.append(charger_id)
            if len(ordered) >= limit:
                return ordered
    return ordered


def simulate_realized_route(
    instance: Instance,
    realized_nodes: Sequence[str],
    policy: ChargingPolicy,
    expected_customers: Optional[Sequence[str]] = None,
) -> RouteResult:
    if not realized_nodes:
        return RouteResult(
            customers=list(expected_customers or []),
            realized_nodes=[],
            events=[],
            feasible=True,
            total_distance=0.0,
            charger_visits=0,
            objective=ObjectiveValue(0.0, 0, 0),
        )

    route_nodes = list(realized_nodes)
    if route_nodes[0] != instance.depot or route_nodes[-1] != instance.depot:
        return RouteResult(
            customers=list(expected_customers or []),
            realized_nodes=route_nodes,
            events=[],
            feasible=False,
            total_distance=float("inf"),
            charger_visits=10**9,
            objective=ObjectiveValue.infinity(),
            message="Route must start and end at the depot.",
        )

    battery_capacity = instance.battery_capacity
    charge_rate = instance.inv_ref_rate

    visited_customers: List[str] = []
    events = [
        StopEvent(
            node_id=instance.depot,
            node_type="DEPOT",
            arrival_time=0.0,
            service_start_time=0.0,
            departure_time=0.0,
            soc_arrival=battery_capacity,
            soc_departure=battery_capacity,
            charge_amount=0.0,
        )
    ]

    soc = battery_capacity
    elapsed = 0.0
    load = 0.0
    total_distance = 0.0
    charger_visits = 0
    total_charge_time = 0.0

    for idx in range(len(route_nodes) - 1):
        source_id = route_nodes[idx]
        target_id = route_nodes[idx + 1]
        source_node = instance.nodes[source_id]
        target_node = instance.nodes[target_id]

        current_event = events[-1]
        if source_node.ntype == "DEPOT":
            soc = battery_capacity
            current_event = StopEvent(
                node_id=current_event.node_id,
                node_type=current_event.node_type,
                arrival_time=current_event.arrival_time,
                service_start_time=current_event.service_start_time,
                departure_time=elapsed,
                soc_arrival=current_event.soc_arrival,
                soc_departure=soc,
                charge_amount=0.0,
            )
            events[-1] = current_event
        elif source_node.ntype == "CHARGER":
            required_energy = _required_energy_to_next_recharge(instance, route_nodes, idx)
            target_soc = policy.recharge_target(soc, required_energy, battery_capacity)
            if target_soc + EPSILON < required_energy:
                return RouteResult(
                    customers=list(expected_customers or []),
                    realized_nodes=route_nodes,
                    events=events,
                    feasible=False,
                    total_distance=float("inf"),
                    charger_visits=10**9,
                    objective=ObjectiveValue.infinity(),
                    message=f"Policy could not supply enough energy at charger {source_id}.",
                )

            target_soc = max(target_soc, soc)
            charge_amount = max(0.0, target_soc - soc)
            charge_time = charge_rate * charge_amount
            elapsed += charge_time
            total_charge_time += charge_time
            soc = target_soc
            current_event = StopEvent(
                node_id=current_event.node_id,
                node_type=current_event.node_type,
                arrival_time=current_event.arrival_time,
                service_start_time=current_event.service_start_time,
                departure_time=elapsed,
                soc_arrival=current_event.soc_arrival,
                soc_departure=soc,
                charge_amount=charge_amount,
            )
            events[-1] = current_event

        required = energy_needed(instance, source_id, target_id)
        if soc + EPSILON < required:
            return RouteResult(
                customers=list(expected_customers or []),
                realized_nodes=route_nodes,
                events=events,
                feasible=False,
                total_distance=float("inf"),
                charger_visits=10**9,
                objective=ObjectiveValue.infinity(),
                message=f"Insufficient energy on arc {source_id} -> {target_id}.",
            )

        soc -= required
        travel_distance = instance.distances[(source_id, target_id)]
        elapsed += travel_distance
        total_distance += travel_distance

        arrival_time = elapsed
        service_start = max(arrival_time, target_node.ready)
        if service_start > target_node.due + EPSILON:
            return RouteResult(
                customers=list(expected_customers or []),
                realized_nodes=route_nodes,
                events=events,
                feasible=False,
                total_distance=float("inf"),
                charger_visits=10**9,
                objective=ObjectiveValue.infinity(),
                message=f"Time window violated at {target_id}.",
            )

        elapsed = service_start + target_node.service

        if target_node.ntype == "CUSTOMER":
            visited_customers.append(target_id)
            load += target_node.demand
            if load > instance.cargo_capacity + EPSILON:
                return RouteResult(
                    customers=list(expected_customers or []),
                    realized_nodes=route_nodes,
                    events=events,
                    feasible=False,
                    total_distance=float("inf"),
                    charger_visits=10**9,
                    objective=ObjectiveValue.infinity(),
                    message=f"Capacity exceeded at {target_id}.",
                )

        if target_node.ntype == "CHARGER":
            charger_visits += 1

        events.append(
            StopEvent(
                node_id=target_id,
                node_type=target_node.ntype,
                arrival_time=arrival_time,
                service_start_time=service_start,
                departure_time=elapsed,
                soc_arrival=soc,
                soc_departure=soc,
                charge_amount=0.0,
            )
        )

    if expected_customers is not None and visited_customers != list(expected_customers):
        return RouteResult(
            customers=list(expected_customers),
            realized_nodes=route_nodes,
            events=events,
            feasible=False,
            total_distance=float("inf"),
            charger_visits=10**9,
            objective=ObjectiveValue.infinity(),
            message="Realized route did not preserve the requested customer order.",
        )

    customers = list(expected_customers or visited_customers)
    vehicle_count = 1 if customers else 0
    return RouteResult(
        customers=customers,
        realized_nodes=route_nodes,
        events=events,
        feasible=True,
        total_distance=total_distance,
        charger_visits=charger_visits,
        objective=ObjectiveValue(total_distance, vehicle_count, charger_visits),
        total_charge_time=total_charge_time,
        completion_time=elapsed,
    )


def evaluate_realized_route(
    instance: Instance,
    realized_nodes: Sequence[str],
    policy: ChargingPolicy,
    expected_customers: Optional[Sequence[str]] = None,
) -> RouteResult:
    return simulate_realized_route(instance, realized_nodes, policy, expected_customers)

def _greedy_insert_chargers(
    instance: Instance,
    customers: Sequence[str],
    policy: ChargingPolicy,
) -> Optional[List[str]]:
    """Greedily insert charging stations into a customer sequence."""
    realized: List[str] = [instance.depot]
    soc = instance.battery_capacity
    targets = list(customers) + [instance.depot]
    idx = 0
    recently_inserted: set = set()

    while idx < len(targets):
        current = realized[-1]
        target = targets[idx]
        needed = energy_needed(instance, current, target)

        if soc + EPSILON >= needed:
            realized.append(target)
            soc -= needed
            idx += 1
            recently_inserted.clear()
        else:
            best_charger = None
            best_cost = float("inf")

            for charger_id in instance.chargers:
                if charger_id == current or charger_id in recently_inserted:
                    continue
                e_to_c = energy_needed(instance, current, charger_id)
                if e_to_c > soc + EPSILON:
                    continue
                soc_at_c = soc - e_to_c
                e_c_to_t = energy_needed(instance, charger_id, target)
                charged = policy.recharge_target(soc_at_c, e_c_to_t, instance.battery_capacity)
                charged = max(charged, soc_at_c)
                if charged + EPSILON < e_c_to_t:
                    continue
                cost = (instance.distances[(current, charger_id)]
                        + instance.distances[(charger_id, target)])
                if cost < best_cost:
                    best_cost = cost
                    best_charger = charger_id

            if best_charger is None:
                return None

            recently_inserted.add(best_charger)
            e_to_c = energy_needed(instance, current, best_charger)
            soc_at_c = soc - e_to_c
            e_c_to_t = energy_needed(instance, best_charger, target)
            charged = policy.recharge_target(soc_at_c, e_c_to_t, instance.battery_capacity)
            soc = max(charged, soc_at_c)
            realized.append(best_charger)

    return realized


def _label_setting_insert_chargers(
    instance: Instance,
    customers: Sequence[str],
    policy: ChargingPolicy,
    max_zero_hops: int = 3,
    deadline: Optional[float] = None,
) -> Optional[Tuple[str, ...]]:
    """Exact charger insertion via label-setting. Used as a fallback when greedy fails."""
    n_customers = len(customers)
    prefix = _customer_prefix_distances(instance, customers)
    priority_queue: List[Tuple[float, float, int, float, int, _Label]] = []
    state_labels: Dict[Tuple[int, str, int], List[_Label]] = {}
    counter = 0

    start_label = _Label(
        total_distance=0.0,
        time_value=0.0,
        soc=instance.battery_capacity,
        charger_visits=0,
        index=0,
        node_id=instance.depot,
        zero_hops=0,
        route_nodes=(instance.depot,),
    )
    state_labels[(0, instance.depot, 0)] = [start_label]
    heappush(priority_queue, (0.0, 0.0, 0, -instance.battery_capacity, counter, start_label))

    final_labels: List[_Label] = []

    while priority_queue:
        _check_deadline(deadline)
        _, _, _, _, _, label = heappop(priority_queue)
        existing = state_labels.get((label.index, label.node_id, label.zero_hops), [])
        if label not in existing:
            continue

        if label.index == n_customers and label.node_id == instance.depot:
            final_labels.append(label)
            continue

        transition_candidates: List[Tuple[int, str, bool]] = []

        if label.index < n_customers and label.zero_hops < max_zero_hops:
            next_customer = customers[label.index]
            zero_move_candidates = _charger_candidates(instance, [next_customer, label.node_id], limit=5)
            for charger_id in zero_move_candidates:
                if charger_id != label.node_id:
                    transition_candidates.append((label.index - 1, charger_id, True))

        if label.index == n_customers:
            if label.node_id != instance.depot and label.zero_hops < max_zero_hops:
                transition_candidates.append((n_customers - 1, instance.depot, True))
        else:
            for end_index in range(label.index, n_customers):
                if end_index == n_customers - 1:
                    transition_candidates.append((end_index, instance.depot, False))
                anchor_nodes = [customers[end_index]]
                if end_index + 1 < n_customers:
                    anchor_nodes.append(customers[end_index + 1])
                anchor_nodes.append(label.node_id)
                for charger_id in _charger_candidates(instance, anchor_nodes, limit=5):
                    transition_candidates.append((end_index, charger_id, False))

        for end_index, end_node, zero_move in transition_candidates:
            _check_deadline(deadline)
            segment_distance = _segment_distance(
                instance,
                prefix,
                customers,
                label.node_id,
                label.index,
                end_index,
                end_node,
            )
            segment_energy = instance.energy_rate * segment_distance
            if segment_energy > instance.battery_capacity + EPSILON:
                continue

            departure_soc = label.soc
            departure_time = label.time_value

            if label.node_id == instance.depot:
                departure_soc = instance.battery_capacity
            elif instance.nodes[label.node_id].ntype == "CHARGER":
                departure_soc = policy.recharge_target(label.soc, segment_energy, instance.battery_capacity)
                if departure_soc + EPSILON < segment_energy:
                    continue
                departure_soc = max(departure_soc, label.soc)
                departure_time += instance.inv_ref_rate * max(0.0, departure_soc - label.soc)

            soc = departure_soc
            current_time = departure_time
            current_node = label.node_id

            if zero_move:
                segment_nodes = [end_node]
            else:
                segment_nodes = list(customers[label.index : end_index + 1]) + [end_node]

            feasible = True
            for target_id in segment_nodes:
                _check_deadline(deadline)
                required = energy_needed(instance, current_node, target_id)
                if soc + EPSILON < required:
                    feasible = False
                    break

                soc -= required
                current_time += instance.distances[(current_node, target_id)]
                target = instance.nodes[target_id]
                if current_time < target.ready:
                    current_time = target.ready
                if current_time > target.due + EPSILON:
                    feasible = False
                    break
                current_time += target.service
                current_node = target_id

            if not feasible:
                continue

            next_index = label.index if zero_move else end_index + 1
            next_zero_hops = label.zero_hops + 1 if zero_move else 0
            next_charger_visits = label.charger_visits + (1 if end_node in instance.chargers else 0)
            next_route_nodes = label.route_nodes + tuple(segment_nodes)

            next_label = _Label(
                total_distance=label.total_distance + segment_distance,
                time_value=current_time,
                soc=soc,
                charger_visits=next_charger_visits,
                index=next_index,
                node_id=end_node,
                zero_hops=next_zero_hops,
                route_nodes=next_route_nodes,
            )

            state_key = (next_index, end_node, next_zero_hops)
            bucket = state_labels.setdefault(state_key, [])
            if any(_dominates(existing_label, next_label) for existing_label in bucket):
                continue

            bucket[:] = [existing_label for existing_label in bucket if not _dominates(next_label, existing_label)]
            bucket.append(next_label)

            counter += 1
            heappush(
                priority_queue,
                (
                    next_label.total_distance,
                    next_label.time_value,
                    next_label.charger_visits,
                    -next_label.soc,
                    counter,
                    next_label,
                ),
            )

    if not final_labels:
        return None

    best_label = min(final_labels, key=lambda item: (item.total_distance, item.charger_visits, item.time_value))
    return best_label.route_nodes


def evaluate_route(
    instance: Instance,
    route: Route,
    policy: ChargingPolicy,
    cache: Optional[Dict[Tuple[Tuple[str, ...], str], RouteResult]] = None,
    max_zero_hops: int = 3,
    deadline: Optional[float] = None,
) -> RouteResult:
    _check_deadline(deadline)
    customers = tuple(customer for customer in route.customers if customer)
    if not customers:
        return RouteResult(
            customers=[],
            realized_nodes=[instance.depot, instance.depot],
            events=[],
            feasible=True,
            total_distance=0.0,
            charger_visits=0,
            objective=ObjectiveValue(0.0, 0, 0),
        )

    if len(set(customers)) != len(customers):
        return RouteResult(
            customers=list(customers),
            realized_nodes=[],
            events=[],
            feasible=False,
            total_distance=float("inf"),
            charger_visits=10**9,
            objective=ObjectiveValue.infinity(),
            message="Repeated customers are not allowed inside one route.",
        )

    cache_key = (customers, policy.policy_id())
    if cache is not None and cache_key in cache:
        return cache[cache_key]

    realized = _greedy_insert_chargers(instance, customers, policy)
    if realized is not None:
        greedy_result = simulate_realized_route(instance, realized, policy, customers)
        if greedy_result.feasible:
            if cache is not None:
                cache[cache_key] = greedy_result
            return greedy_result

    # Fallback: exact label-setting search when greedy fails (rare, hard instances).
    fallback_nodes = _label_setting_insert_chargers(
        instance, customers, policy, max_zero_hops=max_zero_hops, deadline=deadline
    )
    if fallback_nodes is None:
        result = RouteResult(
            customers=list(customers),
            realized_nodes=[],
            events=[],
            feasible=False,
            total_distance=float("inf"),
            charger_visits=10**9,
            objective=ObjectiveValue.infinity(),
            message="No feasible charger insertion found for the customer order.",
        )
        if cache is not None:
            cache[cache_key] = result
        return result

    result = simulate_realized_route(instance, fallback_nodes, policy, customers)
    if cache is not None:
        cache[cache_key] = result
    return result

def evaluate_solution(
    instance: Instance,
    solution: Solution,
    policy: ChargingPolicy,
    cache: Optional[Dict[Tuple[Tuple[str, ...], str], RouteResult]] = None,
    deadline: Optional[float] = None,
) -> Solution:
    cleaned_routes = [Route([customer for customer in route.customers if customer]) for route in solution.routes]
    cleaned_routes = [route for route in cleaned_routes if route.customers]
    solution.routes = cleaned_routes

    seen: set[str] = set()
    route_results: List[RouteResult] = []
    total_distance = 0.0
    total_chargers = 0

    for route in solution.routes:
        _check_deadline(deadline)
        result = evaluate_route(instance, route, policy, cache=cache, deadline=deadline)
        route_results.append(result)
        if not result.feasible:
            solution.feasible = False
            solution.objective = ObjectiveValue.infinity()
            solution.route_results = route_results
            return solution

        for customer in route.customers:
            if customer in seen:
                solution.feasible = False
                solution.objective = ObjectiveValue.infinity()
                solution.route_results = route_results
                return solution
            seen.add(customer)

        total_distance += result.total_distance
        total_chargers += result.charger_visits

    if seen != set(instance.customers):
        solution.feasible = False
        solution.objective = ObjectiveValue.infinity()
        solution.route_results = route_results
        return solution

    solution.feasible = True
    solution.route_results = route_results
    solution.objective = ObjectiveValue(total_distance, len(solution.routes), total_chargers)
    return solution


def build_initial_solution(
    instance: Instance,
    policy: ChargingPolicy,
    cache: Optional[Dict[Tuple[Tuple[str, ...], str], RouteResult]] = None,
    deadline: Optional[float] = None,
) -> Solution:
    solution = Solution(routes=[Route([customer]) for customer in instance.customers])
    evaluated = evaluate_solution(instance, solution, policy, cache=cache, deadline=deadline)
    if not evaluated.feasible:
        raise RuntimeError("Could not build a feasible singleton-route initial solution.")
    return evaluated


def random_destroy(solution: Solution, remove_count: int) -> Tuple[Solution, List[str]]:
    customer_positions: List[Tuple[int, int, str]] = []
    for route_index, route in enumerate(solution.routes):
        for position, customer in enumerate(route.customers):
            customer_positions.append((route_index, position, customer))

    random.shuffle(customer_positions)
    to_remove = {customer for _, _, customer in customer_positions[:remove_count]}

    new_routes: List[Route] = []
    for route in solution.routes:
        remaining = [customer for customer in route.customers if customer not in to_remove]
        if remaining:
            new_routes.append(Route(remaining))

    return Solution(routes=new_routes), sorted(to_remove)


def greedy_repair(
    solution: Solution,
    removed_customers: Sequence[str],
    instance: Instance,
    policy: ChargingPolicy,
    cache: Optional[Dict[Tuple[Tuple[str, ...], str], RouteResult]] = None,
    max_candidate_routes: int = 12,
    deadline: Optional[float] = None,
) -> Solution:
    for customer in removed_customers:
        _check_deadline(deadline)
        best_delta = float("inf")
        best_route_index = None
        best_position = None

        ranked_route_indices = sorted(
            range(len(solution.routes)),
            key=lambda idx: min(
                instance.distances[(customer, route_customer)] for route_customer in solution.routes[idx].customers
            )
            if solution.routes[idx].customers
            else 0.0,
        )

        for route_index in ranked_route_indices[:max_candidate_routes]:
            _check_deadline(deadline)
            route = solution.routes[route_index]
            current_result = evaluate_route(instance, route, policy, cache=cache, deadline=deadline)
            if not current_result.feasible:
                continue

            for position in range(len(route.customers) + 1):
                _check_deadline(deadline)
                candidate_route = Route(route.customers[:position] + [customer] + route.customers[position:])
                candidate_result = evaluate_route(instance, candidate_route, policy, cache=cache, deadline=deadline)
                if not candidate_result.feasible:
                    continue

                delta = candidate_result.total_distance - current_result.total_distance
                if delta < best_delta - EPSILON:
                    best_delta = delta
                    best_route_index = route_index
                    best_position = position

        if best_route_index is None:
            new_route = Route([customer])
            if not evaluate_route(instance, new_route, policy, cache=cache, deadline=deadline).feasible:
                raise RuntimeError(f"Could not reinsert customer {customer} in a feasible route.")
            solution.routes.append(new_route)
            continue

        target_route = solution.routes[best_route_index]
        target_route.customers = (
            target_route.customers[:best_position] + [customer] + target_route.customers[best_position:]
        )

    return evaluate_solution(instance, solution, policy, cache=cache, deadline=deadline)


def alns(
    instance: Instance,
    policy: ChargingPolicy,
    time_budget_seconds: float,
    remove_fraction: float = 0.3,
    seed: int = 42,
    progress_callback=None,
) -> Solution:
    if time_budget_seconds is None or time_budget_seconds <= 0:
        raise ValueError("time_budget_seconds must be a positive number.")

    random.seed(seed)
    route_cache: Dict[Tuple[Tuple[str, ...], str], RouteResult] = {}

    clock = time.perf_counter
    start = clock()
    deadline = start + time_budget_seconds

    current = build_initial_solution(instance, policy, cache=route_cache)
    best = copy_solution(current)
    best_iteration = 0
    iterations_completed = 0
    stop_reason = "time_budget"

    if progress_callback:
        progress_callback(
            {
                "event": "initial_solution",
                "objective": best.objective,
                "routes": len(best.routes),
                "elapsed": clock() - start,
            }
        )

    if clock() >= deadline:
        best.search_stats = SearchStats(
            iterations_completed=iterations_completed,
            best_iteration=best_iteration,
            stop_reason=stop_reason,
            configured_time_budget_seconds=time_budget_seconds,
        )
        return best

    initial_temperature = max(10.0, current.objective.total_distance * 0.02)
    final_temperature = max(1e-3, initial_temperature * 0.01)
    temperature = initial_temperature

    iteration = 0
    while True:
        if clock() >= deadline:
            break

        try:
            candidate = copy_solution(current)
            remove_count = max(1, int(remove_fraction * len(instance.customers)))
            candidate, removed = random_destroy(candidate, remove_count)
            candidate = greedy_repair(candidate, removed, instance, policy, cache=route_cache, deadline=deadline)
        except TimeBudgetExceeded:
            break
        iteration += 1
        iterations_completed = iteration

        if candidate.feasible:
            distance_delta = candidate.objective.total_distance - current.objective.total_distance
            accept = False

            if better_objective(candidate.objective, current.objective):
                accept = True
            elif abs(distance_delta) <= EPSILON and candidate.objective < current.objective:
                accept = True
            elif distance_delta > 0:
                acceptance_prob = math.exp(-distance_delta / max(temperature, 1e-6))
                accept = random.random() < acceptance_prob

            if accept:
                current = candidate

            if better_objective(current.objective, best.objective):
                best = copy_solution(current)
                best_iteration = iteration
                if progress_callback:
                    progress_callback(
                        {
                            "event": "new_best",
                            "iteration": iteration,
                            "objective": best.objective,
                            "routes": len(best.routes),
                            "elapsed": clock() - start,
                        }
                    )

        elapsed_after = clock() - start
        progress = min(1.0, elapsed_after / time_budget_seconds)
        temperature = _scheduled_temperature(initial_temperature, final_temperature, progress)

        if progress_callback and iteration % 10 == 0:
            progress_callback(
                {
                    "event": "iteration",
                    "iteration": iteration,
                    "objective": current.objective,
                    "best_objective": best.objective,
                    "routes": len(current.routes),
                    "elapsed": elapsed_after,
                }
            )

        if clock() >= deadline:
            break

    best.search_stats = SearchStats(
        iterations_completed=iterations_completed,
        best_iteration=best_iteration,
        stop_reason=stop_reason,
        configured_time_budget_seconds=time_budget_seconds,
    )

    return best


def solve_run(config: RunConfig, progress_callback=None) -> RunResult:
    policy = make_policy_from_spec(config.policy_spec)
    instance = read_schneider_instance(config.instance_path)
    started = time.time()

    try:
        solution = alns(
            instance,
            policy,
            time_budget_seconds=config.time_budget_seconds,
            remove_fraction=config.remove_fraction,
            seed=config.seed,
            progress_callback=progress_callback,
        )
        elapsed = time.time() - started
        return RunResult(
            config=config,
            solution=solution,
            elapsed_seconds=elapsed,
            status="ok" if solution.feasible else "infeasible",
            iterations_completed=solution.search_stats.iterations_completed,
            best_iteration=solution.search_stats.best_iteration,
            stop_reason=solution.search_stats.stop_reason,
        )
    except Exception as exc:
        elapsed = time.time() - started
        return RunResult(
            config=config,
            solution=Solution(routes=[]),
            elapsed_seconds=elapsed,
            status="error",
            error_message=str(exc),
            stop_reason="error",
        )


def print_solution(solution: Solution) -> None:
    print(f"Feasible: {solution.feasible}")
    print(f"Objective: {objective_string(solution.objective)}")
    for route_index, result in enumerate(solution.route_results, start=1):
        print(f"Route {route_index}: {' -> '.join(result.realized_nodes)}")


def plot_solution(instance: Instance, solution: Solution, title: str = "EVRPTW Solution"):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 9))

    used_labels = set()
    for nid, node in instance.nodes.items():
        if node.ntype == "DEPOT":
            color, marker, label, size = "blue", "s", "Depot", 120
        elif node.ntype == "CUSTOMER":
            color, marker, label, size = "gold", "o", "Customer", 60
        else:
            color, marker, label, size = "red", "^", "Charger", 90

        if label not in used_labels:
            ax.scatter(node.x, node.y, c=color, marker=marker, s=size, label=label)
            used_labels.add(label)
        else:
            ax.scatter(node.x, node.y, c=color, marker=marker, s=size)

        if node.ntype != "CUSTOMER":
            ax.text(node.x + 0.5, node.y + 0.5, nid, fontsize=8)

    colors = plt.cm.tab20.colors
    for index, route_result in enumerate(solution.route_results):
        color = colors[index % len(colors)]
        for edge_index in range(len(route_result.realized_nodes) - 1):
            source_id = route_result.realized_nodes[edge_index]
            target_id = route_result.realized_nodes[edge_index + 1]
            source = instance.nodes[source_id]
            target = instance.nodes[target_id]
            ax.plot([source.x, target.x], [source.y, target.y], "-", color=color, linewidth=2)

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def _cli_policy(policy_name: str, levels: Sequence[float]) -> ChargingPolicy:
    if policy_name == "full":
        return FullChargePolicy()
    if policy_name == "fixed":
        return FixedPartialPolicy(levels)
    if policy_name == "continuous":
        return ContinuousPartialPolicy()
    raise ValueError(f"Unknown policy: {policy_name}")


def _main() -> None:
    parser = argparse.ArgumentParser(description="Run the EVRPTW ALNS solver on a Schneider instance.")
    parser.add_argument("--instance", default="data_schneider/c101_21.txt")
    parser.add_argument("--policy", choices=["full", "fixed", "continuous"], default="full")
    parser.add_argument("--time-budget-seconds", type=float, default=600.0)
    parser.add_argument("--remove-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fixed-levels", default="0.5,0.8,1.0")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    fixed_levels = [float(value.strip()) for value in args.fixed_levels.split(",") if value.strip()]
    instance = read_schneider_instance(args.instance)
    policy = _cli_policy(args.policy, fixed_levels)

    def cli_progress(payload: Dict[str, object]) -> None:
        event = payload["event"]
        if event == "new_best":
            print(f"New best at iter {payload['iteration']}: {objective_string(payload['objective'])}")
        elif event == "iteration":
            print(
                f"Iter {payload['iteration']}: current={objective_string(payload['objective'])}, "
                f"best={objective_string(payload['best_objective'])}"
            )
        elif event == "initial_solution":
            print(f"Initial solution: {objective_string(payload['objective'])}")

    started = time.time()
    solution = alns(
        instance,
        policy,
        time_budget_seconds=args.time_budget_seconds,
        remove_fraction=args.remove_fraction,
        seed=args.seed,
        progress_callback=cli_progress,
    )
    elapsed = time.time() - started

    print_solution(solution)
    print(
        "Search stats: "
        f"iterations_completed={solution.search_stats.iterations_completed}, "
        f"best_iteration={solution.search_stats.best_iteration}, "
        f"stop_reason={solution.search_stats.stop_reason}, "
        f"elapsed_seconds={elapsed:.2f}"
    )
    if args.plot:
        plot_solution(instance, solution, title=f"{policy.spec.label} on {Path(args.instance).name}")
        import matplotlib.pyplot as plt

        plt.show()


if __name__ == "__main__":
    _main()
