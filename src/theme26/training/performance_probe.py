"""Per-phase wall-time and allocated-GPU-memory logging for bounded pilots."""

from __future__ import annotations

import os
import time
from pathlib import Path

import torch
import yaml


class PhaseProbe:
    def __init__(self, *, epoch: int, phase: str, expected_steps: int,
                 device: torch.device, output_dir: Path,
                 log_interval_steps: int | None = None) -> None:
        if phase not in ("train", "validation") or expected_steps < 2:
            raise ValueError("phase must be train/validation with at least two steps")
        if log_interval_steps is not None and log_interval_steps < 1:
            raise ValueError("log_interval_steps must be positive")
        self.epoch = epoch
        self.phase = phase
        self.expected_steps = expected_steps
        self.device = device
        self.output_dir = output_dir
        self.log_interval_steps = log_interval_steps
        self.events: list[dict] = []
        self.started = 0.0
        self.last_samples = 0

    def _gpu_peak(self) -> int | None:
        return torch.cuda.max_memory_allocated(self.device) if self.device.type == "cuda" else None

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _record(self, steps: int, samples: int) -> None:
        self._sync()
        elapsed = time.perf_counter() - self.started
        event = {"epoch": self.epoch, "phase": self.phase, "step": steps,
                 "expected_steps": self.expected_steps, "samples": samples,
                 "elapsed_seconds": elapsed,
                 "steps_per_second": steps / elapsed if steps else 0.0,
                 "samples_per_second": samples / elapsed if steps else 0.0,
                 "peak_gpu_allocated_bytes": self._gpu_peak()}
        self.events.append(event)
        path = self.output_dir / "progress.yaml"
        temporary = path.with_suffix(".yaml.tmp")
        existing = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else []
        existing.append(event)
        temporary.write_text(yaml.safe_dump(existing, sort_keys=False), encoding="utf-8")
        os.replace(temporary, path)
        print(f"epoch={self.epoch:03d} phase={self.phase} "
              f"step={steps:03d}/{self.expected_steps:03d} "
              f"elapsed_seconds={elapsed:.1f} "
              f"steps_per_second={event['steps_per_second']:.3f} "
              f"samples_per_second={event['samples_per_second']:.3f} "
              f"peak_gpu_allocated_bytes={event['peak_gpu_allocated_bytes']}", flush=True)

    def start(self) -> None:
        self._sync()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        self.started = time.perf_counter()
        self._record(0, 0)

    def on_step(self, steps: int, samples: int) -> None:
        self.last_samples = samples
        if (steps == self.expected_steps or
                (self.log_interval_steps is None and steps == self.expected_steps // 2) or
                (self.log_interval_steps is not None and steps % self.log_interval_steps == 0)):
            self._record(steps, samples)

    def finish(self, actual_steps: int) -> dict:
        if actual_steps != self.expected_steps:
            raise ValueError(f"{self.phase} produced {actual_steps} steps; "
                             f"expected {self.expected_steps}")
        self._sync()
        elapsed = time.perf_counter() - self.started
        return {"steps": actual_steps, "samples": self.last_samples,
                "elapsed_seconds": elapsed,
                "steps_per_second": actual_steps / elapsed,
                "samples_per_second": self.last_samples / elapsed,
                "peak_gpu_allocated_bytes": self._gpu_peak()}
