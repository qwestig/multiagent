from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .reporting import build_results_rows
from .schemas import DatasetCase, ExperimentRun, StoragePaths
from .utils import make_slug


class ProjectStorage:
    def __init__(self, paths: StoragePaths) -> None:
        self.paths = paths
        self.ensure_directories()

    @classmethod
    def default(cls, root: Path) -> "ProjectStorage":
        return cls(
            StoragePaths(
                root=root,
                dataset_path=root / "data" / "dataset.jsonl",
                runs_dir=root / "runs",
                reports_dir=root / "reports",
                anti_error_path=root / "runs" / "anti_errors.json",
            )
        )

    def ensure_directories(self) -> None:
        self.paths.root.mkdir(parents=True, exist_ok=True)
        self.paths.runs_dir.mkdir(parents=True, exist_ok=True)
        self.paths.reports_dir.mkdir(parents=True, exist_ok=True)
        if not self.paths.anti_error_path.exists():
            self.paths.anti_error_path.write_text("{}", encoding="utf-8")

    def load_dataset(self) -> list[DatasetCase]:
        cases: list[DatasetCase] = []
        if not self.paths.dataset_path.exists():
            return cases
        for line in self.paths.dataset_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            cases.append(DatasetCase.from_dict(json.loads(line)))
        return cases

    def save_run(self, run: ExperimentRun) -> Path:
        filename = f"{run.created_at.replace(':', '-').replace('T', '_')}-{make_slug(run.case_id)}.json"
        target = self.paths.runs_dir / filename
        target.write_text(
            json.dumps(run.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return target

    def load_run(self, path: Path) -> ExperimentRun:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return ExperimentRun.from_dict(payload)

    def list_runs(self, limit: int = 50) -> list[Path]:
        files = [
            path
            for path in self.paths.runs_dir.glob("*.json")
            if path.name != self.paths.anti_error_path.name
        ]
        files = sorted(files, reverse=True)
        return files[:limit]

    def export_results_csv(self, runs: list[ExperimentRun], name: str | None = None) -> Path:
        import csv

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        filename = name or f"results-{timestamp}.csv"
        target = self.paths.reports_dir / filename
        rows = build_results_rows(runs)
        if not rows:
            target.write_text("", encoding="utf-8")
            return target
        with target.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        return target

    def save_report(self, markdown: str, name: str | None = None) -> Path:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        filename = name or f"report-{timestamp}.md"
        target = self.paths.reports_dir / filename
        target.write_text(markdown, encoding="utf-8")
        return target

    def load_anti_errors(self) -> dict[str, list[str]]:
        return json.loads(self.paths.anti_error_path.read_text(encoding="utf-8"))

    def remember_anti_error(self, bucket: str, rule: str) -> None:
        if not rule.strip():
            return
        payload = self.load_anti_errors()
        bucket_rules = payload.setdefault(bucket, [])
        if rule not in bucket_rules:
            bucket_rules.append(rule)
        self.paths.anti_error_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def recent_anti_errors(self, bucket: str, limit: int = 5) -> list[str]:
        payload = self.load_anti_errors()
        return payload.get(bucket, [])[-limit:]
