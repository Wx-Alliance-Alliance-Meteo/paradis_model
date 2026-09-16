from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.utilities import rank_zero_only
from omegaconf import DictConfig, OmegaConf


def setup_system(cfg: DictConfig) -> None:
    """Set seeds and numerical precision."""

    if "init" in cfg and "seed" in cfg.init:
        L.seed_everything(cfg.init.seed, workers=True)

    if cfg.compute.use_amp:
        torch.set_float32_matmul_precision("medium")
    else:
        torch.set_float32_matmul_precision("high")

    if not cfg.forecast.enable:
        assert not (
            cfg.training.max_steps < 0 and cfg.training.max_epochs < 0
        ), "Please indicate max_epochs or max_steps"


@rank_zero_only
def save_train_config(log_dir: str, cfg: DictConfig) -> None:
    """
    Save an immutable configuration snapshot for each PBS attempt.

    Directory structure:

        <log_dir>/
            config.yaml
            config_latest.yaml
            code_snapshot/
            attempts/
                attempt01-<job-id>/
                    config.yaml
                    config_resolved.yaml
                    metadata.yaml
                    git_commit.txt
                    git_status.txt
                    git_diff.patch
    """

    # Each allocated node initially launches a Python process. This prevents
    # the parent process on NODE_RANK > 0 from writing duplicate snapshots.
    if int(os.environ.get("NODE_RANK", "0")) != 0:
        return

    experiment_dir = Path(log_dir).resolve()
    experiment_dir.mkdir(parents=True, exist_ok=True)

    attempt = _read_attempt_number()
    job_id = _safe_path_component(
        os.environ.get("PIPELINE_PBS_JOBID", "no-pbs-job")
    )

    attempt_name = f"attempt{attempt:02d}-{job_id}"
    attempt_dir = experiment_dir / "attempts" / attempt_name
    attempt_dir.mkdir(parents=True, exist_ok=True)

    unresolved_yaml = OmegaConf.to_yaml(
        cfg,
        resolve=False,
        sort_keys=False,
    )

    try:
        resolved_yaml = OmegaConf.to_yaml(
            cfg,
            resolve=True,
            sort_keys=False,
        )
    except Exception as exc:
        resolved_yaml = (
            "# The fully resolved configuration could not be produced.\n"
            f"# Error: {type(exc).__name__}: {exc}\n\n"
            + unresolved_yaml
        )

    # Canonical initial configuration. It is deliberately never overwritten.
    _write_once(
        experiment_dir / "config.yaml",
        unresolved_yaml,
    )

    # Immutable configuration for this particular PBS attempt.
    _write_once(
        attempt_dir / "config.yaml",
        unresolved_yaml,
    )

    _write_once(
        attempt_dir / "config_resolved.yaml",
        resolved_yaml,
    )

    # Convenience copy. This one is intentionally updated each attempt.
    _atomic_write(
        experiment_dir / "config_latest.yaml",
        unresolved_yaml,
    )

    metadata = {
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_name": cfg.training.get("experiment_name", None),
        "stage_key": os.environ.get("PIPELINE_STAGE_KEY", None),
        "attempt": attempt,
        "pbs_job_id": os.environ.get("PIPELINE_PBS_JOBID", None),
        "node_rank": os.environ.get("NODE_RANK", None),
        "restart": cfg.init.get("restart", None),
        "checkpoint_path": cfg.init.get("checkpoint_path", None),
        "max_steps": cfg.training.get("max_steps", None),
        "learning_rate": cfg.training.optimizer.get("lr", None),
        "forecast_steps": cfg.model.get("forecast_steps", None),
    }

    metadata_yaml = OmegaConf.to_yaml(
        OmegaConf.create(metadata),
        resolve=True,
        sort_keys=False,
    )

    _write_once(
        attempt_dir / "metadata.yaml",
        metadata_yaml,
    )

    # Record the exact git state for every attempt.
    _save_git_metadata(attempt_dir)

    # A complete tracked-code snapshot is saved once for the experiment-stage.
    # Per-attempt git metadata records later changes without duplicating the
    # entire repository on every 12-hour continuation.
    code_snapshot_dir = experiment_dir / "code_snapshot"

    if not code_snapshot_dir.exists():
        _save_code_snapshot(code_snapshot_dir)

    print(f"Saved attempt configuration: {attempt_dir}")
    print(f"Canonical configuration:     {experiment_dir / 'config.yaml'}")
    print(f"Latest configuration:        {experiment_dir / 'config_latest.yaml'}")


def _read_attempt_number() -> int:
    value = os.environ.get("PIPELINE_ATTEMPT", "0")

    try:
        return int(value)
    except ValueError as exc:
        raise RuntimeError(
            f"Invalid PIPELINE_ATTEMPT value: {value!r}"
        ) from exc


def _safe_path_component(value: str) -> str:
    """Convert a PBS identifier into a safe filename component."""

    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return cleaned.strip("._") or "unknown"


def _write_once(path: Path, content: str) -> None:
    """
    Write a file once.

    If the file already exists with identical content, this is harmless.
    If it exists with different content, refuse to overwrite it.
    """

    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        existing = path.read_text(encoding="utf-8")

        if existing == content:
            return

        raise RuntimeError(
            f"Refusing to overwrite an existing snapshot with different "
            f"content: {path}"
        )

    _atomic_write(path, content)


def _atomic_write(path: Path, content: str) -> None:
    """Write through a temporary file and atomically replace the target."""

    path.parent.mkdir(parents=True, exist_ok=True)

    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
        text=True,
    )

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())

        os.replace(temporary_name, path)

    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass

        raise


def _run_git(
    project_root: Path,
    *arguments: str,
    check: bool = True,
) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=check,
    )

    return result.stdout


def _get_project_root() -> Path:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=True,
    )

    return Path(result.stdout.strip()).resolve()


def _save_git_metadata(attempt_dir: Path) -> None:
    """Save commit, branch, status and working-tree changes."""

    try:
        project_root = _get_project_root()

        commit = _run_git(
            project_root,
            "rev-parse",
            "HEAD",
        ).strip()

        branch = _run_git(
            project_root,
            "rev-parse",
            "--abbrev-ref",
            "HEAD",
        ).strip()

        status = _run_git(
            project_root,
            "status",
            "--short",
            "--branch",
        )

        tracked_diff = _run_git(
            project_root,
            "diff",
            "--binary",
            "HEAD",
        )

        untracked_files = _run_git(
            project_root,
            "ls-files",
            "--others",
            "--exclude-standard",
        )

        _write_once(
            attempt_dir / "git_commit.txt",
            f"{commit}\n",
        )

        _write_once(
            attempt_dir / "git_branch.txt",
            f"{branch}\n",
        )

        _write_once(
            attempt_dir / "git_status.txt",
            status,
        )

        _write_once(
            attempt_dir / "git_diff.patch",
            tracked_diff,
        )

        _write_once(
            attempt_dir / "git_untracked_files.txt",
            untracked_files,
        )

    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        warning = (
            "Git metadata could not be saved.\n"
            f"{type(exc).__name__}: {exc}\n"
        )

        _write_once(
            attempt_dir / "git_error.txt",
            warning,
        )


def _save_code_snapshot(code_snapshot_dir: Path) -> None:
    """Save the current contents of all tracked Git files."""

    try:
        project_root = _get_project_root()

        tracked_output = _run_git(
            project_root,
            "ls-files",
        )

        tracked_files = [
            line
            for line in tracked_output.splitlines()
            if line
        ]

        code_snapshot_dir.mkdir(parents=True, exist_ok=False)

        for tracked_file in tracked_files:
            source = project_root / tracked_file
            destination = code_snapshot_dir / tracked_file

            # A tracked file may have been deleted in the working tree.
            if not source.is_file():
                continue

            destination.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            shutil.copy2(source, destination)

    except FileExistsError:
        # Another rank or earlier invocation already created it.
        return

    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise RuntimeError(
            f"Failed to save code snapshot: {exc}"
        ) from exc