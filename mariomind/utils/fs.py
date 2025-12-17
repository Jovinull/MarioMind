from __future__ import annotations

from pathlib import Path

from mariomind.utils.paths import (
    assets_dir,
    checkpoints_dir,
    docs_assets_dir,
    logs_dir,
    media_dir,
    models_dir,
    runs_dir,
)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def ensure_project_layout() -> None:
    """
    Garante que as pastas base existam, sem precisar de gitkeep.
    """
    ensure_dir(runs_dir())
    ensure_dir(assets_dir())
    ensure_dir(checkpoints_dir())
    ensure_dir(docs_assets_dir())


def ensure_run_layout(run_id: str) -> dict[str, Path]:
    """
    Cria e devolve o layout padrão de uma run:
    runs/<run_id>/{models,logs,media}
    """
    ensure_project_layout()

    md = models_dir(run_id)
    ld = logs_dir(run_id)
    vd = media_dir(run_id)

    ensure_dir(md)
    ensure_dir(ld)
    ensure_dir(vd)

    return {"models": md, "logs": ld, "media": vd}
