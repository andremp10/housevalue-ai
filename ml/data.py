"""Descoberta, verificação e carregamento dos dados House Prices.

Regras do PRD (§6.1 e §17):
- descobrir o ZIP sem presumir nome/caminho;
- nunca executar arquivos do ZIP, nunca sobrescrever o original;
- extrair uma cópia de trabalho em data/raw/ sem sobrescrever arquivos existentes;
- registrar caminho, tamanho e SHA-256.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from ml import config

# Divergências conhecidas entre o nome usado no dicionário e o nome da coluna no CSV.
# São verificadas (não presumidas) em inspect_data.py; aqui apenas mapeamos.
DICTIONARY_NAME_ALIASES = {"Bedroom": "BedroomAbvGr", "Kitchen": "KitchenAbvGr"}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _iter_zip_candidates(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in config.SEARCH_EXCLUDE_DIRS]
        for name in filenames:
            if name.lower().endswith(".zip"):
                yield Path(dirpath) / name


def zip_contains_house_prices(path: Path) -> bool:
    """Um ZIP é considerado candidato se algum CSV interno tiver a coluna SalePrice."""
    try:
        with zipfile.ZipFile(path) as zf:
            for info in zf.infolist():
                if info.filename.lower().endswith(".csv"):
                    with zf.open(info) as fh:
                        header = fh.readline().decode("utf-8", errors="replace")
                    if "SalePrice" in [h.strip().strip('"') for h in header.strip().split(",")]:
                        return True
    except (zipfile.BadZipFile, OSError):
        return False
    return False


@dataclass
class ZipDiscovery:
    searched: list[str] = field(default_factory=list)
    candidates: list[str] = field(default_factory=list)
    selected: str | None = None
    decision: str = ""


def discover_zip() -> ZipDiscovery:
    """Procura o ZIP da base House Prices.

    Ordem: variável de ambiente HOUSEVALUE_ZIP → workspace (recursivo) →
    pasta Downloads do usuário (somente nível raiz, usada como fallback
    documentado porque o ZIP original foi encontrado lá nesta máquina).
    """
    result = ZipDiscovery()
    env = os.environ.get("HOUSEVALUE_ZIP")
    if env:
        result.searched.append(f"env:HOUSEVALUE_ZIP={env}")
        p = Path(env)
        if p.is_file() and zip_contains_house_prices(p):
            result.candidates.append(str(p))
            result.selected = str(p)
            result.decision = "ZIP indicado pela variável de ambiente HOUSEVALUE_ZIP."
            return result

    result.searched.append(f"workspace:{config.PROJECT_ROOT}")
    ws = [p for p in _iter_zip_candidates(config.PROJECT_ROOT) if zip_contains_house_prices(p)]
    result.candidates.extend(str(p) for p in ws)
    if ws:
        result.selected = str(sorted(ws)[0])
        result.decision = "ZIP encontrado dentro do workspace."
        return result

    downloads = Path.home() / "Downloads"
    result.searched.append(f"downloads:{downloads}")
    if downloads.is_dir():
        dl = [p for p in downloads.glob("*.zip") if zip_contains_house_prices(p)]
        result.candidates.extend(str(p) for p in dl)
        if dl:
            # Preferir o nome oficial da competição, se houver mais de um.
            dl.sort(key=lambda p: ("house-prices" not in p.name.lower(), p.name))
            result.selected = str(dl[0])
            result.decision = (
                "Nenhum ZIP no workspace; ZIP com a base House Prices encontrado em Downloads. "
                "O workspace contém os arquivos já extraídos (verificados por SHA-256)."
            )
            return result

    result.decision = "Nenhum ZIP encontrado; serão usados os arquivos extraídos presentes no workspace."
    return result


def build_source_manifest() -> dict:
    """Localiza a fonte, registra hashes e extrai a cópia de trabalho em data/raw/."""
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    disc = discover_zip()
    manifest: dict = {
        "zip_discovery": disc.__dict__,
        "zip": None,
        "zip_members": [],
        "workspace_extracted_files": [],
        "working_copy": [],
    }

    members: dict[str, bytes] = {}
    if disc.selected:
        zp = Path(disc.selected)
        manifest["zip"] = {
            "file_name": zp.name,
            "location": "workspace" if config.PROJECT_ROOT in zp.parents else "user Downloads folder",
            "size_bytes": zp.stat().st_size,
            "sha256": sha256_file(zp),
        }
        with zipfile.ZipFile(zp) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                data = zf.read(info)  # apenas leitura de bytes, nada é executado
                members[Path(info.filename).name] = data
                manifest["zip_members"].append(
                    {
                        "name": info.filename,
                        "size_bytes": info.file_size,
                        "compressed_bytes": info.compress_size,
                        "sha256": sha256_bytes(data),
                    }
                )

    # Arquivos já extraídos na raiz do workspace (não são modificados).
    for name in config.EXPECTED_FILES:
        p = config.PROJECT_ROOT / name
        if p.is_file():
            h = sha256_file(p)
            manifest["workspace_extracted_files"].append(
                {
                    "name": name,
                    "size_bytes": p.stat().st_size,
                    "sha256": h,
                    "identical_to_zip_member": (sha256_bytes(members[name]) == h) if name in members else None,
                }
            )

    # Cópia de trabalho: a partir do ZIP; se não houver ZIP, a partir dos extraídos.
    for name in config.EXPECTED_FILES:
        dest = config.RAW_DIR / name
        if name in members:
            data, origin = members[name], "zip"
        elif (config.PROJECT_ROOT / name).is_file():
            data, origin = (config.PROJECT_ROOT / name).read_bytes(), "workspace_root"
        else:
            continue
        h = sha256_bytes(data)
        if dest.exists():
            action = "kept_existing_identical" if sha256_file(dest) == h else "kept_existing_DIFFERENT"
        else:
            dest.write_bytes(data)
            action = "extracted"
        manifest["working_copy"].append(
            {"name": name, "path": f"data/raw/{name}", "origin": origin, "sha256": h, "action": action}
        )

    config.SOURCE_MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def _raw_path(name: str) -> Path:
    p = config.RAW_DIR / name
    if not p.is_file():
        raise FileNotFoundError(f"{p} não encontrado. Execute `python -m ml.inspect_data` primeiro.")
    return p


def read_csv_bytes(path: Path) -> pd.DataFrame:
    """Lê o CSV a partir de bytes (evita problemas de caminho com acentos/OneDrive)."""
    with open(path, "rb") as fh:
        buf = io.BytesIO(fh.read())
    return pd.read_csv(buf, **config.CSV_READ_KWARGS)


def load_train() -> pd.DataFrame:
    return read_csv_bytes(_raw_path(config.TRAIN_FILE))


def load_test() -> pd.DataFrame:
    return read_csv_bytes(_raw_path(config.TEST_FILE))


def load_raw_strings(name: str) -> pd.DataFrame:
    """Leitura sem nenhuma conversão de ausentes, para auditar os tokens brutos."""
    with open(_raw_path(name), "rb") as fh:
        buf = io.BytesIO(fh.read())
    return pd.read_csv(buf, dtype=str, keep_default_na=False)


_COL_RE = re.compile(r"^([A-Za-z0-9]+):\s*(.*)$")


def parse_data_dictionary(text: str | None = None) -> dict:
    """Converte data_description.txt em {coluna: {description, categories{code: label}}}."""
    if text is None:
        text = _raw_path(config.DICTIONARY_FILE).read_text(encoding="utf-8", errors="replace")
    out: dict[str, dict] = {}
    current = None
    for line in text.splitlines():
        if not line.strip():
            continue
        m = _COL_RE.match(line)
        if m and not line[0].isspace():
            name = m.group(1)
            current = {"dictionary_name": name, "description": m.group(2).strip(), "categories": {}}
            out[DICTIONARY_NAME_ALIASES.get(name, name)] = current
            continue
        if current is not None and line[0].isspace():
            parts = line.strip().split("\t", 1)
            code = parts[0].strip()
            label = parts[1].strip() if len(parts) > 1 else ""
            if code:
                current["categories"][code] = label
    return out


def structural_na_columns(dictionary: dict) -> dict[str, str]:
    """Colunas cujo dicionário define o código NA como inexistência do elemento."""
    return {
        col: info["categories"]["NA"]
        for col, info in dictionary.items()
        if "NA" in info["categories"]
    }
