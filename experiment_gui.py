import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from experiment_runner import build_run_configs, run_batch
from main_4 import DEFAULT_FIXED_LEVELS, read_schneider_instance


class ExperimentGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("EVRPTW Experiment Platform")
        self.root.geometry("1500x900")

        self.queue: queue.Queue = queue.Queue()
        self.worker: threading.Thread | None = None
        self.current_summary = None
        self.run_results = {}

        self._build_layout()
        self.root.after(200, self._poll_queue)

    def _build_layout(self) -> None:
        container = ttk.Frame(self.root, padding=12)
        container.pack(fill=tk.BOTH, expand=True)
        container.columnconfigure(0, weight=1)
        container.columnconfigure(1, weight=2)
        container.rowconfigure(0, weight=0)
        container.rowconfigure(1, weight=1)

        self.setup_frame = ttk.LabelFrame(container, text="Run Setup", padding=10)
        self.setup_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 8), pady=(0, 8))

        self.progress_frame = ttk.LabelFrame(container, text="Progress", padding=10)
        self.progress_frame.grid(row=0, column=1, sticky="nsew", pady=(0, 8))

        self.results_frame = ttk.LabelFrame(container, text="Results Table", padding=10)
        self.results_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 8))

        self.charts_frame = ttk.LabelFrame(container, text="Charts / Viewer", padding=10)
        self.charts_frame.grid(row=1, column=1, sticky="nsew")

        self.results_frame.rowconfigure(0, weight=1)
        self.results_frame.columnconfigure(0, weight=1)
        self.charts_frame.rowconfigure(1, weight=1)
        self.charts_frame.columnconfigure(0, weight=1)

        self._build_setup_controls()
        self._build_progress_controls()
        self._build_results_table()
        self._build_chart_viewer()

    def _build_setup_controls(self) -> None:
        self.preset_var = tk.StringVar(value="pilot")
        self.time_budget_var = tk.StringVar(value="600")
        self.remove_fraction_var = tk.StringVar(value="0.2")
        self.seeds_var = tk.StringVar(value="42")
        self.fixed_levels_var = tk.StringVar(value="0.5,0.8,1.0")
        self.output_root_var = tk.StringVar(value="results")
        self.include_validation_var = tk.BooleanVar(value=False)
        self.policy_full_var = tk.BooleanVar(value=True)
        self.policy_fixed_var = tk.BooleanVar(value=True)
        self.policy_cont_var = tk.BooleanVar(value=True)

        fields = [
            ("Preset", ttk.Combobox(self.setup_frame, textvariable=self.preset_var, values=["sanity", "pilot", "paper_subset", "paper_full", "validation"], state="readonly")),
            ("Time budget (s)", ttk.Entry(self.setup_frame, textvariable=self.time_budget_var)),
            ("Remove fraction", ttk.Entry(self.setup_frame, textvariable=self.remove_fraction_var)),
            ("Seeds", ttk.Entry(self.setup_frame, textvariable=self.seeds_var)),
            ("Fixed levels", ttk.Entry(self.setup_frame, textvariable=self.fixed_levels_var)),
            ("Output root", ttk.Entry(self.setup_frame, textvariable=self.output_root_var)),
        ]

        for row_index, (label, widget) in enumerate(fields):
            ttk.Label(self.setup_frame, text=label).grid(row=row_index, column=0, sticky="w", pady=2)
            widget.grid(row=row_index, column=1, sticky="ew", pady=2)

        self.setup_frame.columnconfigure(1, weight=1)

        ttk.Label(self.setup_frame, text="Policies").grid(row=6, column=0, sticky="w", pady=(8, 2))
        policy_box = ttk.Frame(self.setup_frame)
        policy_box.grid(row=6, column=1, sticky="w")
        ttk.Checkbutton(policy_box, text="Full", variable=self.policy_full_var).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Checkbutton(policy_box, text="Fixed", variable=self.policy_fixed_var).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Checkbutton(policy_box, text="Continuous", variable=self.policy_cont_var).pack(side=tk.LEFT)

        ttk.Checkbutton(
            self.setup_frame,
            text="Include exact-model validation for full charge",
            variable=self.include_validation_var,
        ).grid(row=7, column=0, columnspan=2, sticky="w", pady=(8, 4))

        button_row = ttk.Frame(self.setup_frame)
        button_row.grid(row=8, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        ttk.Button(button_row, text="Run Batch", command=self._start_batch).pack(side=tk.LEFT)
        ttk.Button(button_row, text="Clear Results", command=self._clear_results).pack(side=tk.LEFT, padx=8)

    def _build_progress_controls(self) -> None:
        self.status_var = tk.StringVar(value="Idle")
        self.output_dir_var = tk.StringVar(value="")
        ttk.Label(self.progress_frame, textvariable=self.status_var).pack(anchor="w")
        ttk.Label(self.progress_frame, textvariable=self.output_dir_var, foreground="#555555").pack(anchor="w", pady=(4, 8))

        self.log_text = tk.Text(self.progress_frame, height=12, wrap="word")
        self.log_text.pack(fill=tk.BOTH, expand=True)
        self.log_text.configure(state=tk.DISABLED)

    def _build_results_table(self) -> None:
        columns = ("run_id", "instance", "policy", "feasible", "distance", "vehicles", "chargers", "runtime", "iters", "stop")
        self.tree = ttk.Treeview(self.results_frame, columns=columns, show="headings", selectmode="browse")
        self.tree.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(self.results_frame, orient=tk.VERTICAL, command=self.tree.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=scrollbar.set)

        headings = {
            "run_id": "Run ID",
            "instance": "Instance",
            "policy": "Policy",
            "feasible": "Feasible",
            "distance": "Distance",
            "vehicles": "Vehicles",
            "chargers": "Chargers",
            "runtime": "Runtime (s)",
            "iters": "Iters",
            "stop": "Stop",
        }
        for column, heading in headings.items():
            self.tree.heading(column, text=heading, command=lambda col=column: self._sort_tree(col, False))
            self.tree.column(column, width=120, anchor="center")
        self.tree.bind("<<TreeviewSelect>>", lambda _: self._plot_selected_route())

    def _build_chart_viewer(self) -> None:
        button_bar = ttk.Frame(self.charts_frame)
        button_bar.grid(row=0, column=0, sticky="w", pady=(0, 8))
        ttk.Button(button_bar, text="Plot Selected Route", command=self._plot_selected_route).pack(side=tk.LEFT)
        ttk.Button(button_bar, text="Policy Distance Summary", command=self._plot_distance_summary).pack(side=tk.LEFT, padx=6)
        ttk.Button(button_bar, text="Policy Runtime Summary", command=self._plot_runtime_summary).pack(side=tk.LEFT)

        self.figure = Figure(figsize=(7, 5), dpi=100)
        self.axes = self.figure.add_subplot(111)
        self.axes.set_title("Charts will appear here")
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.charts_frame)
        self.canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew")

    def _selected_policy_names(self) -> list[str]:
        selected = []
        if self.policy_full_var.get():
            selected.append("full")
        if self.policy_fixed_var.get():
            selected.append("fixed")
        if self.policy_cont_var.get():
            selected.append("continuous")
        return selected

    def _start_batch(self) -> None:
        if self.worker and self.worker.is_alive():
            messagebox.showinfo("Experiment runner", "A batch is already running.")
            return

        policy_names = self._selected_policy_names()
        if not policy_names:
            messagebox.showerror("Experiment runner", "Select at least one policy.")
            return

        try:
            time_budget_text = self.time_budget_var.get().strip()
            time_budget_seconds = float(time_budget_text) if time_budget_text else 0.0
            if time_budget_seconds <= 0:
                raise ValueError("Time budget must be a positive number of seconds.")
            remove_fraction = float(self.remove_fraction_var.get())
            seeds = [int(token.strip()) for token in self.seeds_var.get().split(",") if token.strip()]
            fixed_levels = [float(token.strip()) for token in self.fixed_levels_var.get().split(",") if token.strip()]
        except ValueError as exc:
            messagebox.showerror("Invalid input", str(exc))
            return

        configs = build_run_configs(
            preset=self.preset_var.get(),
            time_budget_seconds=time_budget_seconds,
            remove_fraction=remove_fraction,
            seeds=seeds,
            fixed_levels=fixed_levels or DEFAULT_FIXED_LEVELS,
        )

        configs = [config for config in configs if config.policy_spec.name in policy_names]
        self.status_var.set(f"Queued {len(configs)} runs")
        self.output_dir_var.set("")
        self._append_log("Starting batch run")

        def worker() -> None:
            summary = run_batch(
                configs,
                preset_name=self.preset_var.get(),
                output_root=self.output_root_var.get(),
                include_validation=self.include_validation_var.get(),
                progress_callback=self.queue.put,
            )
            self.queue.put({"type": "batch_complete", "summary": summary})

        self.worker = threading.Thread(target=worker, daemon=True)
        self.worker.start()

    def _append_log(self, message: str) -> None:
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def _clear_results(self) -> None:
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.run_results.clear()
        self.current_summary = None
        self.status_var.set("Idle")
        self.output_dir_var.set("")
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.delete("1.0", tk.END)
        self.log_text.configure(state=tk.DISABLED)
        self.axes.clear()
        self.axes.set_title("Charts will appear here")
        self.canvas.draw_idle()

    def _poll_queue(self) -> None:
        try:
            while True:
                payload = self.queue.get_nowait()
                self._handle_payload(payload)
        except queue.Empty:
            pass
        self.root.after(200, self._poll_queue)

    def _handle_payload(self, payload) -> None:
        payload_type = payload.get("type")
        if payload_type == "log":
            self.status_var.set(payload["message"])
            self._append_log(payload["message"])
        elif payload_type == "solver_progress":
            if payload["event"] == "new_best":
                self._append_log(
                    f"{payload['instance']} / {payload['policy']} iter {payload['iteration']}: "
                    f"{payload['objective']}"
                )
        elif payload_type == "run_complete":
            run_result = payload["run_result"]
            self._append_log(payload["message"])
            self._insert_run_result(run_result)
        elif payload_type == "batch_complete":
            self.current_summary = payload["summary"]
            self.status_var.set("Batch completed")
            self.output_dir_var.set(f"Output: {self.current_summary.output_dir}")
            self._append_log(f"Saved results to {self.current_summary.output_dir}")
            self._plot_distance_summary()

    def _insert_run_result(self, run_result) -> None:
        run_id = f"{Path(run_result.config.instance_path).stem}__{run_result.config.policy_spec.name}__seed{run_result.config.seed}"
        if self.tree.exists(run_id):
            self.tree.delete(run_id)
        self.run_results[run_id] = run_result
        values = (
            run_id,
            Path(run_result.config.instance_path).name,
            run_result.config.policy_spec.name,
            str(run_result.solution.feasible),
            f"{run_result.solution.objective.total_distance:.2f}" if run_result.solution.feasible else "inf",
            run_result.solution.objective.vehicle_count,
            run_result.solution.objective.charger_visits,
            f"{run_result.elapsed_seconds:.2f}",
            run_result.iterations_completed,
            run_result.stop_reason,
        )
        self.tree.insert("", tk.END, iid=run_id, values=values)

    def _sort_tree(self, column: str, reverse: bool) -> None:
        rows = [(self.tree.set(item, column), item) for item in self.tree.get_children("")]

        def convert(value: str):
            try:
                return float(value)
            except ValueError:
                return value

        rows.sort(key=lambda item: convert(item[0]), reverse=reverse)
        for index, (_, item) in enumerate(rows):
            self.tree.move(item, "", index)
        self.tree.heading(column, command=lambda: self._sort_tree(column, not reverse))

    def _plot_selected_route(self) -> None:
        selection = self.tree.selection()
        if not selection:
            return
        run_result = self.run_results.get(selection[0])
        if not run_result or not run_result.solution.feasible:
            return

        instance = read_schneider_instance(run_result.config.instance_path)
        self.axes.clear()

        used_labels = set()
        for node_id, node in instance.nodes.items():
            if node.ntype == "DEPOT":
                color, marker, label, size = "blue", "s", "Depot", 100
            elif node.ntype == "CUSTOMER":
                color, marker, label, size = "gold", "o", "Customer", 40
            else:
                color, marker, label, size = "red", "^", "Charger", 70

            if label not in used_labels:
                self.axes.scatter(node.x, node.y, c=color, marker=marker, s=size, label=label)
                used_labels.add(label)
            else:
                self.axes.scatter(node.x, node.y, c=color, marker=marker, s=size)

        colors = ["#204a87", "#4e9a06", "#cc0000", "#75507b", "#f57900", "#ce5c00"]
        for idx, route_result in enumerate(run_result.solution.route_results):
            color = colors[idx % len(colors)]
            for edge_index in range(len(route_result.realized_nodes) - 1):
                source = instance.nodes[route_result.realized_nodes[edge_index]]
                target = instance.nodes[route_result.realized_nodes[edge_index + 1]]
                self.axes.plot([source.x, target.x], [source.y, target.y], color=color, linewidth=2)

        self.axes.set_title(f"{Path(run_result.config.instance_path).stem} - {run_result.config.policy_spec.label}")
        self.axes.set_xlabel("X")
        self.axes.set_ylabel("Y")
        self.axes.grid(True, alpha=0.3)
        self.axes.legend()
        self.canvas.draw_idle()

    def _plot_distance_summary(self) -> None:
        if not self.current_summary:
            return
        self.axes.clear()
        rows = self.current_summary.summary_overall
        policies = [row["policy"] for row in rows]
        values = [float(row["mean_best_distance"]) for row in rows]
        self.axes.bar(policies, values, color=["#3465a4", "#4e9a06", "#cc0000"][: len(policies)])
        self.axes.set_title("Mean best distance by policy")
        self.axes.set_ylabel("Distance")
        self.canvas.draw_idle()

    def _plot_runtime_summary(self) -> None:
        if not self.current_summary:
            return
        self.axes.clear()
        rows = self.current_summary.summary_overall
        policies = [row["policy"] for row in rows]
        values = [float(row["mean_runtime"]) for row in rows]
        self.axes.bar(policies, values, color=["#75507b", "#f57900", "#73d216"][: len(policies)])
        self.axes.set_title("Mean runtime by policy")
        self.axes.set_ylabel("Seconds")
        self.canvas.draw_idle()


def main() -> None:
    root = tk.Tk()
    ExperimentGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
