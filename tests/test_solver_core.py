import json
import tempfile
import unittest
from pathlib import Path

import main_4 as solver
from experiment_runner import build_run_configs, preset_instance_paths, run_batch


class SolverCoreTests(unittest.TestCase):
    def test_schneider_listing_ignores_readme(self):
        paths = solver.list_schneider_instances("data_schneider")
        names = {Path(path).name for path in paths}
        self.assertIn("c101_21.txt", names)
        self.assertNotIn("readme.txt", {name.lower() for name in names})

    def test_continuous_partial_route_fix(self):
        instance = solver.read_schneider_instance("data_schneider/r101_21.txt")
        policy = solver.ContinuousPartialPolicy()
        result = solver.evaluate_realized_route(
            instance,
            ["D0", "S3", "C64", "S5", "D0"],
            policy,
            expected_customers=["C64"],
        )
        self.assertTrue(result.feasible, result.message)
        self.assertAlmostEqual(result.total_distance, 110.33799495619846, places=6)
        self.assertEqual(result.charger_visits, 2)

    def test_initial_solution_succeeds_on_full_schneider_batch(self):
        policies = [
            solver.FullChargePolicy(),
            solver.FixedPartialPolicy([0.5, 0.8, 1.0]),
            solver.ContinuousPartialPolicy(),
        ]

        for path in solver.list_schneider_instances("data_schneider"):
            instance = solver.read_schneider_instance(path)
            for policy in policies:
                with self.subTest(instance=Path(path).name, policy=policy.spec.name):
                    solution = solver.build_initial_solution(instance, policy, cache={})
                    self.assertTrue(solution.feasible)
                    self.assertEqual(solution.objective.vehicle_count, len(instance.customers))

    def test_batch_runner_exports_consistent_route_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            configs = build_run_configs(
                preset="sanity",
                time_budget_seconds=2.0,
                remove_fraction=0.2,
                seeds=[42],
                policy_specs=[solver.ContinuousPartialPolicy().spec],
                fixed_levels=[0.5, 0.8, 1.0],
            )
            configs = configs[:1]
            summary = run_batch(configs, preset_name="sanity_test", output_root=tmpdir, include_validation=False)

            output_dir = Path(summary.output_dir)
            self.assertTrue((output_dir / "runs.csv").exists())
            self.assertTrue((output_dir / "routes.json").exists())
            self.assertTrue((output_dir / "summary_by_instance_policy.csv").exists())
            self.assertTrue((output_dir / "summary_overall.csv").exists())
            self.assertTrue((output_dir / "table_summary.tex").exists())

            payload = json.loads((output_dir / "routes.json").read_text(encoding="utf-8"))
            self.assertEqual(len(payload), 1)
            run_payload = payload[0]
            total_distance = 0.0
            total_chargers = 0
            for route in run_payload["routes"]:
                route_distance = 0.0
                events = route["events"]
                realized_nodes = route["realized_nodes"]
                self.assertEqual(events[0]["node_id"], realized_nodes[0])
                self.assertEqual(events[-1]["node_id"], realized_nodes[-1])
                for event in events:
                    if event["node_type"] == "CHARGER":
                        total_chargers += 1
                total_distance += route["total_distance"]

            self.assertAlmostEqual(total_distance, run_payload["objective"]["total_distance"], places=6)
            self.assertEqual(total_chargers, run_payload["objective"]["charger_visits"])

    def test_validation_preset_contains_small_instances(self):
        validation_paths = preset_instance_paths("validation", data_dir="data_schneider")
        self.assertGreaterEqual(len(validation_paths), 1)
        for path in validation_paths:
            self.assertIn(Path(path).name, {"c101C5.txt", "r102C10.txt", "rc108C5.txt"})


if __name__ == "__main__":
    unittest.main()
