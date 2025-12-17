from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable


def project_root() -> Path:
    # mariomind/utils/paths.py -> parents[0]=utils, [1]=mariomind, [2]=root
    return Path(__file__).resolve().parents[2]


def runs_dir() -> Path:
    return project_root() / "runs"


def assets_dir() -> Path:
    return project_root() / "assets"


def checkpoints_dir() -> Path:
    return assets_dir() / "checkpoints"


def docs_assets_dir() -> Path:
    return project_root() / "docs" / "assets"


def make_run_id(prefix: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}"


def run_dir(run_id: str) -> Path:
    return runs_dir() / run_id


def models_dir(run_id: str) -> Path:
    return run_dir(run_id) / "models"


def logs_dir(run_id: str) -> Path:
    return run_dir(run_id) / "logs"


def media_dir(run_id: str) -> Path:
    return run_dir(run_id) / "media"


def resolve_existing_path(candidates: Iterable[Path]) -> Path | None:
    for p in candidates:
        if p.exists():
            return p
    return None


def resolve_checkpoint(filename: str, extra_fallbacks: list[str] | None = None) -> Path:
    """
    Resolve o caminho de um checkpoint de forma robusta.

    Ordem de busca:
    1) assets/checkpoints/<filename>
    2) raiz do projeto/<filename> (compatibilidade com seu layout antigo)
    3) quaisquer fallbacks adicionais informados
    """
    root = project_root()
    fallbacks = extra_fallbacks or []

    candidates = [
        checkpoints_dir() / filename,
        root / filename,
    ] + [root / f for f in fallbacks] + [checkpoints_dir() / f for f in fallbacks]

    found = resolve_existing_path(candidates)
    return found if found is not None else (checkpoints_dir() / filename)
