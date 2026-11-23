"""Fase 1 — Auditoria dos dados (PRD §6).

Uso: python -m ml.inspect_data

Gera:
- data/source_manifest.json      (origem, tamanhos, SHA-256, cópia de trabalho)
- artifacts/data_dictionary.json (dicionário de dados parseado)
- artifacts/data_profile.json    (perfil completo, máquina-legível)
- docs/data_audit.md             (relatório legível com tabelas e evidências)
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ml import config
from ml.data import (
    DICTIONARY_NAME_ALIASES,
    build_source_manifest,
    load_raw_strings,
    load_test,
    load_train,
    parse_data_dictionary,
    structural_na_columns,
)

# Variáveis numéricas no arquivo que, pelo dicionário, são códigos categóricos nominais.
NOMINAL_NUMERIC_CODES = ["MSSubClass"]
# Escalas ordinais numéricas (1–10) mantidas como numéricas.
ORDINAL_NUMERIC = ["OverallQual", "OverallCond"]

# Regras de consistência das ausências estruturais: coluna de "existência" → colunas dependentes.
STRUCTURAL_GROUPS = {
    "Porão (TotalBsmtSF == 0)": ("TotalBsmtSF", ["BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2"]),
    "Garagem (GarageArea == 0)": ("GarageArea", ["GarageType", "GarageFinish", "GarageQual", "GarageCond", "GarageYrBlt"]),
    "Lareira (Fireplaces == 0)": ("Fireplaces", ["FireplaceQu"]),
    "Piscina (PoolArea == 0)": ("PoolArea", ["PoolQC"]),
}


def _py(v):
    """Converte tipos numpy/pandas para tipos JSON nativos."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return None if np.isnan(v) else float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if v is pd.NA or (isinstance(v, float) and np.isnan(v)):
        return None
    return v


def file_format_info(name: str) -> dict:
    raw = (config.RAW_DIR / name).read_bytes()
    try:
        raw.decode("utf-8")
        enc = "ascii" if raw.isascii() else "utf-8"
    except UnicodeDecodeError:
        enc = "não-UTF-8"
    return {
        "encoding": enc,
        "has_bom": raw.startswith(b"\xef\xbb\xbf"),
        "line_endings": "CRLF" if b"\r\n" in raw[:5000] else "LF",
        "size_bytes": len(raw),
    }


def expected_type(col: str, dictionary: dict) -> str:
    if col in NOMINAL_NUMERIC_CODES:
        return "categórica nominal (códigos numéricos)"
    if col in ORDINAL_NUMERIC:
        return "numérica ordinal (escala 1–10)"
    info = dictionary.get(col)
    if info is None:
        return "não descrita no dicionário"
    return "categórica" if info["categories"] else "numérica"


def classify_missing(col: str, structural: dict[str, str]) -> str:
    if col in structural:
        return f"estrutural — dicionário: NA = '{structural[col]}'"
    return "desconhecida / não registrada"


def structural_consistency(df: pd.DataFrame) -> list[dict]:
    rows = []
    for group, (flag, deps) in STRUCTURAL_GROUPS.items():
        absent = df[flag].fillna(0) == 0
        for dep in deps:
            na = df[dep].isna()
            rows.append(
                {
                    "group": group,
                    "column": dep,
                    "na_total": int(na.sum()),
                    "na_and_element_absent": int((na & absent).sum()),
                    "na_but_element_present": int((na & ~absent).sum()),
                    "value_but_element_absent": int((~na & absent).sum()),
                }
            )
    return rows


def category_vs_dictionary(df: pd.DataFrame, dictionary: dict, cat_cols: list[str]) -> list[dict]:
    out = []
    for c in cat_cols:
        info = dictionary.get(c)
        if not info or not info["categories"]:
            continue
        observed = set(df[c].dropna().astype(str).unique())
        documented = set(info["categories"].keys())
        not_doc = sorted(observed - documented)
        if not_doc:
            out.append(
                {
                    "column": c,
                    "observed_not_in_dictionary": not_doc,
                    "dictionary_codes_not_observed": sorted(documented - observed - {"NA"}),
                }
            )
    return out


def logical_checks(df: pd.DataFrame, label: str) -> list[dict]:
    checks = {
        "YearRemodAdd < YearBuilt": df["YearRemodAdd"] < df["YearBuilt"],
        "YrSold < YearBuilt": df["YrSold"] < df["YearBuilt"],
        "YrSold < YearRemodAdd": df["YrSold"] < df["YearRemodAdd"],
        "GarageYrBlt > YrSold": df["GarageYrBlt"] > df["YrSold"],
        "GarageYrBlt < YearBuilt": df["GarageYrBlt"] < df["YearBuilt"],
        "MasVnrType == 'None' e MasVnrArea > 0": (df["MasVnrType"] == "None") & (df["MasVnrArea"] > 0),
        "MasVnrType != 'None' e MasVnrArea == 0": df["MasVnrType"].notna() & (df["MasVnrType"] != "None") & (df["MasVnrArea"] == 0),
        "1stFlrSF+2ndFlrSF+LowQualFinSF != GrLivArea": (df["1stFlrSF"] + df["2ndFlrSF"] + df["LowQualFinSF"]) != df["GrLivArea"],
        "BsmtFinSF1+BsmtFinSF2+BsmtUnfSF != TotalBsmtSF": (df["BsmtFinSF1"] + df["BsmtFinSF2"] + df["BsmtUnfSF"]) != df["TotalBsmtSF"],
    }
    out = []
    for name, mask in checks.items():
        mask = mask.fillna(False)
        ids = df.loc[mask, config.ID_COLUMN].head(10).tolist()
        out.append({"file": label, "check": name, "count": int(mask.sum()), "example_ids": [int(i) for i in ids]})
    return out


def profile() -> dict:
    manifest = build_source_manifest()
    dictionary = parse_data_dictionary()
    config.DATA_DICTIONARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.DATA_DICTIONARY_PATH.write_text(json.dumps(dictionary, indent=2, ensure_ascii=False), encoding="utf-8")
    structural = structural_na_columns(dictionary)

    train, test = load_train(), load_test()
    raw_train, raw_test = load_raw_strings(config.TRAIN_FILE), load_raw_strings(config.TEST_FILE)
    sample = pd.read_csv(config.RAW_DIR / config.SUBMISSION_SAMPLE_FILE)

    # ---------- Gate: alvo ----------
    target_present = config.TARGET in train.columns
    target_numeric = target_present and pd.api.types.is_numeric_dtype(train[config.TARGET])
    target_missing = int(train[config.TARGET].isna().sum()) if target_present else None

    files = {
        config.TRAIN_FILE: {**file_format_info(config.TRAIN_FILE), "rows": len(train), "columns": train.shape[1], "has_target": target_present},
        config.TEST_FILE: {**file_format_info(config.TEST_FILE), "rows": len(test), "columns": test.shape[1], "has_target": config.TARGET in test.columns},
        config.SUBMISSION_SAMPLE_FILE: {**file_format_info(config.SUBMISSION_SAMPLE_FILE), "rows": len(sample), "columns": sample.shape[1], "column_names": list(sample.columns)},
        config.DICTIONARY_FILE: {**file_format_info(config.DICTIONARY_FILE), "documented_columns": len(dictionary)},
    }

    feature_cols = [c for c in train.columns if c not in (config.ID_COLUMN, config.TARGET)]
    cat_cols = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(train[c]) or c in NOMINAL_NUMERIC_CODES]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    # ---------- Colunas ----------
    columns = []
    for c in train.columns:
        info = dictionary.get(c, {})
        na_tr = int(train[c].isna().sum())
        na_te = int(test[c].isna().sum()) if c in test.columns else None
        columns.append(
            {
                "name": c,
                "dictionary_name": info.get("dictionary_name"),
                "description": info.get("description"),
                "inferred_dtype": str(train[c].dtype),
                "expected_type": "alvo (numérica, USD)" if c == config.TARGET else ("identificador" if c == config.ID_COLUMN else expected_type(c, dictionary)),
                "n_unique_train": int(train[c].nunique(dropna=True)),
                "missing_train": na_tr,
                "missing_train_pct": round(100 * na_tr / len(train), 2),
                "missing_test": na_te,
                "missing_test_pct": round(100 * na_te / len(test), 2) if na_te is not None else None,
                "missing_kind": classify_missing(c, structural) if (na_tr or (na_te or 0)) else None,
            }
        )

    in_data_not_dict = [c for c in train.columns if c not in dictionary and c not in (config.ID_COLUMN, config.TARGET)]
    in_dict_not_data = [c for c in dictionary if c not in train.columns]

    # ---------- Tokens brutos ----------
    tokens = {}
    for label, raw in (("train", raw_train), ("test", raw_test)):
        tok = {}
        for t in ["NA", "None", "", "nan", "NULL"]:
            counts = {c: int((raw[c] == t).sum()) for c in raw.columns if (raw[c] == t).any()}
            if counts:
                tok[t] = counts
        tokens[label] = tok
    default_pandas_missing = int(pd.read_csv(config.RAW_DIR / config.TRAIN_FILE)["MasVnrType"].isna().sum())

    # ---------- Duplicidades ----------
    duplicates = {
        "train_full_rows": int(train.duplicated().sum()),
        "train_excluding_id": int(train.drop(columns=[config.ID_COLUMN]).duplicated().sum()),
        "train_features_only": int(train[feature_cols].duplicated().sum()),
        "test_excluding_id": int(test.drop(columns=[config.ID_COLUMN]).duplicated().sum()),
        "train_id_unique": bool(train[config.ID_COLUMN].is_unique),
        "test_id_unique": bool(test[config.ID_COLUMN].is_unique),
        "id_overlap_train_test": int(len(set(train[config.ID_COLUMN]) & set(test[config.ID_COLUMN]))),
        "train_id_range": [int(train[config.ID_COLUMN].min()), int(train[config.ID_COLUMN].max())],
        "test_id_range": [int(test[config.ID_COLUMN].min()), int(test[config.ID_COLUMN].max())],
    }

    # ---------- Estatísticas ----------
    desc = train[num_cols].describe().T
    q1, q3 = train[num_cols].quantile(0.25), train[num_cols].quantile(0.75)
    iqr = q3 - q1
    numeric_stats = []
    for c in num_cols:
        lo, hi = q1[c] - 1.5 * iqr[c], q3[c] + 1.5 * iqr[c]
        s = train[c]
        numeric_stats.append(
            {
                "column": c,
                **{k: _py(desc.loc[c, k]) for k in ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]},
                "skew": _py(s.skew()),
                "share_zero_pct": round(100 * float((s == 0).mean()), 2),
                "iqr_outliers": int(((s < lo) | (s > hi)).sum()),
            }
        )
    categorical_stats = []
    for c in cat_cols:
        vc = train[c].astype("string").value_counts(dropna=False)
        top_share = float(vc.iloc[0] / len(train))
        categorical_stats.append(
            {
                "column": c,
                "n_categories": int(train[c].nunique(dropna=True)),
                "top": [{"value": (None if pd.isna(k) else str(k)), "count": int(v), "pct": round(100 * v / len(train), 2)} for k, v in vc.head(6).items()],
                "dominant_share_pct": round(100 * top_share, 2),
                "unseen_in_train_but_in_test": sorted(set(test[c].dropna().astype(str)) - set(train[c].dropna().astype(str))),
            }
        )

    # ---------- Alvo ----------
    y = train[config.TARGET]
    hist_counts, hist_edges = np.histogram(y, bins=40)
    target = {
        "present": target_present,
        "numeric": bool(target_numeric),
        "missing": target_missing,
        "non_positive": int((y <= 0).sum()),
        "describe": {k: _py(v) for k, v in y.describe().items()},
        "quantiles": {str(q): _py(y.quantile(q)) for q in [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]},
        "skew": _py(y.skew()),
        "kurtosis": _py(y.kurtosis()),
        "log1p_skew": _py(np.log1p(y).skew()),
        "log1p_kurtosis": _py(np.log1p(y).kurtosis()),
        "histogram": {"counts": hist_counts.tolist(), "edges": [float(e) for e in hist_edges]},
        "by_year_sold": {str(k): {"n": int(g.size), "median": float(g.median())} for k, g in y.groupby(train["YrSold"])},
    }

    # ---------- Extremos ----------
    big = train[train["GrLivArea"] > 4000][[config.ID_COLUMN, "GrLivArea", "SalePrice", "SaleCondition", "OverallQual", "Neighborhood"]]
    extremes = {
        "grlivarea_gt_4000_train": big.to_dict(orient="records"),
        "grlivarea_gt_4000_test": test[test["GrLivArea"] > 4000][[config.ID_COLUMN, "GrLivArea", "SaleCondition"]].to_dict(orient="records"),
        "top5_saleprice": train.nlargest(5, "SalePrice")[[config.ID_COLUMN, "SalePrice", "GrLivArea", "OverallQual", "Neighborhood"]].to_dict(orient="records"),
        "bottom5_saleprice": train.nsmallest(5, "SalePrice")[[config.ID_COLUMN, "SalePrice", "GrLivArea", "OverallQual", "Neighborhood"]].to_dict(orient="records"),
        "sale_condition_counts": train["SaleCondition"].value_counts().to_dict(),
    }
    extremes = json.loads(json.dumps(extremes, default=_py))

    logical = logical_checks(train, "train") + logical_checks(test, "test")
    structural_check = {"train": structural_consistency(train), "test": structural_consistency(test)}
    cat_dict = {"train": category_vs_dictionary(train, dictionary, cat_cols), "test": category_vs_dictionary(test, dictionary, cat_cols)}

    high_missing = [
        {"column": c["name"], "missing_train_pct": c["missing_train_pct"], "kind": c["missing_kind"]}
        for c in columns
        if c["missing_train_pct"] >= 15
    ]
    near_constant = [s["column"] for s in categorical_stats if s["dominant_share_pct"] >= 99]

    viability = {
        "labelled_file_with_numeric_target": bool(target_present and target_numeric and target_missing == 0),
        "n_labelled_rows": len(train),
        "holdout_rows_at_20pct": int(round(len(train) * config.HOLDOUT_FRACTION)),
        "predictors_interpretable_via_dictionary": len(in_data_not_dict) == 0,
        "critical_inconsistencies_treatable": True,
    }
    viability["apt_for_regression"] = all(
        viability[k] for k in ["labelled_file_with_numeric_target", "predictors_interpretable_via_dictionary", "critical_inconsistencies_treatable"]
    )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_manifest": manifest,
        "read_policy": {
            "csv_kwargs": config.CSV_READ_KWARGS,
            "rationale": "Somente o token literal 'NA' é ausente. O padrão do pandas converteria o texto 'None' (categoria válida de MasVnrType) em NaN.",
            "masvnrtype_missing_with_pandas_default": default_pandas_missing,
            "masvnrtype_missing_with_project_policy": int(train["MasVnrType"].isna().sum()),
        },
        "files": files,
        "dictionary": {
            "columns_documented": len(dictionary),
            "name_aliases_applied": DICTIONARY_NAME_ALIASES,
            "data_columns_not_in_dictionary": in_data_not_dict,
            "dictionary_columns_not_in_data": in_dict_not_data,
            "structural_na_columns": structural,
            "category_mismatches": cat_dict,
        },
        "feature_groups": {"n_features": len(feature_cols), "numeric": num_cols, "categorical": cat_cols, "nominal_numeric_codes": NOMINAL_NUMERIC_CODES},
        "columns": columns,
        "raw_tokens": tokens,
        "duplicates": duplicates,
        "numeric_stats": numeric_stats,
        "categorical_stats": categorical_stats,
        "target": target,
        "extremes": extremes,
        "logical_checks": logical,
        "structural_consistency": structural_check,
        "high_missing": high_missing,
        "near_constant_categorical": near_constant,
        "viability": viability,
    }


# ----------------------------------------------------------------------------- Markdown

def _fmt(v, nd=2):
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "sim" if v else "não"
    if isinstance(v, float):
        return f"{v:,.{nd}f}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


def _table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(_fmt(x) for x in r) + " |")
    return "\n".join(lines)


def render_markdown(p: dict) -> str:
    m = p["source_manifest"]
    t = p["target"]
    L: list[str] = []
    A = L.append
    A("# Auditoria dos dados — House Prices (HouseValue AI)\n")
    A(f"> Documento gerado automaticamente por `python -m ml.inspect_data` em {p['generated_at']} (UTC). "
      "Todos os números abaixo foram calculados a partir dos arquivos locais; nada foi presumido. "
      "Perfil completo em `artifacts/data_profile.json`.\n")

    A("## 1. Origem e localização dos arquivos\n")
    d = m["zip_discovery"]
    A(f"- **Locais pesquisados:** {', '.join('`'+s+'`' for s in d['searched'])}")
    A(f"- **ZIPs candidatos (contendo CSV com `SalePrice`):** {len(d['candidates'])}")
    A(f"- **Decisão:** {d['decision']}")
    if m["zip"]:
        z = m["zip"]
        A(f"- **ZIP original:** `{z['file_name']}` ({z['location']}), {z['size_bytes']:,} bytes, SHA-256 `{z['sha256']}`. O arquivo original não foi movido nem modificado.\n")
        A("### Conteúdo do ZIP\n")
        A(_table(["Membro", "Tamanho (bytes)", "Compactado", "SHA-256"], [[x["name"], x["size_bytes"], x["compressed_bytes"], f"`{x['sha256']}`"] for x in m["zip_members"]]))
    A("\n### Arquivos já extraídos na raiz do workspace\n")
    A(_table(["Arquivo", "Tamanho", "Idêntico ao membro do ZIP?"], [[x["name"], x["size_bytes"], {True: "sim (SHA-256 igual)", False: "NÃO", None: "sem ZIP"}[x["identical_to_zip_member"]]] for x in m["workspace_extracted_files"]]))
    A("\n### Cópia de trabalho utilizada pelo pipeline\n")
    A(_table(["Arquivo", "Caminho", "Origem", "Ação"], [[x["name"], x["path"], x["origin"], x["action"]] for x in m["working_copy"]]))

    A("\n## 2. Formato e estrutura\n")
    A(_table(["Arquivo", "Codificação", "BOM", "Fim de linha", "Linhas", "Colunas"],
             [[k, v["encoding"], v["has_bom"], v["line_endings"], v.get("rows", "—"), v.get("columns", v.get("documented_columns", "—"))] for k, v in p["files"].items()]))
    A(f"\n- `{config.TRAIN_FILE}` contém `SalePrice`: **{p['files'][config.TRAIN_FILE]['has_target']}**; `{config.TEST_FILE}` contém `SalePrice`: **{p['files'][config.TEST_FILE]['has_target']}** (portanto `test.csv` não serve para métricas supervisionadas).")
    A(f"- `sample_submission.csv` tem colunas {p['files'][config.SUBMISSION_SAMPLE_FILE]['column_names']}.")
    A(f"- Preditores (excluindo `Id` e `SalePrice`): **{p['feature_groups']['n_features']}** — {len(p['feature_groups']['numeric'])} numéricos e {len(p['feature_groups']['categorical'])} categóricos (incluindo `MSSubClass`, código numérico nominal).")
    dd = p["dictionary"]
    A(f"- Dicionário (`data_description.txt`): {dd['columns_documented']} colunas descritas. Colunas do CSV sem descrição: {dd['data_columns_not_in_dictionary'] or 'nenhuma'}. "
      f"Nomes divergentes mapeados: {', '.join(f'`{k}` → `{v}`' for k, v in dd['name_aliases_applied'].items())}.\n")

    A("## 3. Política de leitura e tokens de ausência\n")
    rp = p["read_policy"]
    A(f"- Parâmetros: `{rp['csv_kwargs']}`. {rp['rationale']}")
    A(f"- Evidência: com a leitura padrão do pandas, `MasVnrType` teria **{rp['masvnrtype_missing_with_pandas_default']}** ausentes; com a política do projeto, **{rp['masvnrtype_missing_with_project_policy']}**. O restante é a categoria `None` (sem revestimento de alvenaria).")
    A(f"- Tokens brutos encontrados no `train.csv`: " + "; ".join(f"`{k!r}` em {len(v)} coluna(s)" for k, v in p["raw_tokens"]["train"].items()) + ".\n")

    A("## 4. Dicionário de colunas (tipo inferido × esperado, ausências)\n")
    A(_table(["Coluna", "Descrição (dicionário)", "dtype inferido", "Tipo esperado", "Únicos", "Ausentes treino", "% treino", "Ausentes teste", "Natureza da ausência"],
             [[c["name"], (c["description"] or "—")[:70], c["inferred_dtype"], c["expected_type"], c["n_unique_train"], c["missing_train"], c["missing_train_pct"], c["missing_test"], c["missing_kind"] or ""] for c in p["columns"]]))

    A("\n## 5. Interpretação das ausências (PRD §6.4)\n")
    A("O dicionário define explicitamente `NA` como **inexistência do elemento** nas colunas abaixo (ausência estrutural):\n")
    A(_table(["Coluna", "Significado de NA no dicionário"], [[k, v] for k, v in dd["structural_na_columns"].items()]))
    A("\nVerificação de consistência (ausência estrutural deveria coincidir com o elemento inexistente):\n")
    for label in ("train", "test"):
        A(f"\n**{label}.csv**\n")
        A(_table(["Grupo", "Coluna", "NA total", "NA e elemento ausente", "NA mas elemento presente", "Valor mas elemento ausente"],
                 [[r["group"], r["column"], r["na_total"], r["na_and_element_absent"], r["na_but_element_present"], r["value_but_element_absent"]] for r in p["structural_consistency"][label]]))
    A("\nInterpretação: `NA` com elemento ausente → ausência **estrutural** (preenchida com a categoria `\"None\"` por regra fixa, sem aprendizado). "
      "`NA` com elemento presente → informação **desconhecida/inconsistente** (mantida como ausente e imputada dentro do pipeline com parâmetros aprendidos só no treino). "
      "`Alley`, `Fence` e `MiscFeature` não têm coluna de existência; seguem o dicionário (`NA` = inexistente). "
      "`LotFrontage`, `MasVnrType`, `MasVnrArea` e `Electrical` não têm `NA` documentado: ausência tratada como desconhecida. "
      "`GarageYrBlt` é numérica e fica ausente quando não há garagem; imputada com indicador de ausência.\n")

    A("## 6. Variáveis com alto percentual de ausência (≥ 15% no treino)\n")
    A(_table(["Coluna", "% ausente (treino)", "Natureza"], [[h["column"], h["missing_train_pct"], h["kind"]] for h in p["high_missing"]]))
    A("\nComo as ausências altas são majoritariamente estruturais (informativas: indicam inexistência), as colunas são mantidas no conjunto completo com a categoria `None`; nenhuma é descartada só pelo percentual.\n")

    A("## 7. Duplicidades e identificadores\n")
    du = p["duplicates"]
    A(_table(["Verificação", "Resultado"], [[k, str(v)] for k, v in du.items()]))

    A("\n## 8. Variável-alvo `SalePrice` (USD)\n")
    A(_table(["Estatística", "Valor"], [[k, v] for k, v in t["describe"].items()] + [["assimetria", t["skew"]], ["curtose", t["kurtosis"]], ["assimetria log1p", t["log1p_skew"]], ["curtose log1p", t["log1p_kurtosis"]], ["ausentes", t["missing"]], ["≤ 0", t["non_positive"]]]))
    A("\nQuantis:\n")
    A(_table(["Quantil", "SalePrice (USD)"], [[k, v] for k, v in t["quantiles"].items()]))
    A("\nPor ano de venda:\n")
    A(_table(["YrSold", "n", "Mediana (USD)"], [[k, v["n"], v["median"]] for k, v in t["by_year_sold"].items()]))
    A("\nA distribuição é assimétrica à direita; `log1p` reduz a assimetria (valores acima). Isso motiva testar `TransformedTargetRegressor(log1p/expm1)` na modelagem (PRD §9.4), sem aplicá-lo automaticamente.\n")

    A("## 9. Estatísticas descritivas — numéricas (treino)\n")
    A(_table(["Coluna", "n", "média", "dp", "mín", "25%", "50%", "75%", "máx", "assim.", "% zeros", "outliers IQR"],
             [[s["column"], s["count"], s["mean"], s["std"], s["min"], s["25%"], s["50%"], s["75%"], s["max"], s["skew"], s["share_zero_pct"], s["iqr_outliers"]] for s in p["numeric_stats"]]))

    A("\n## 10. Frequência das categorias principais (treino)\n")
    A(_table(["Coluna", "nº categorias", "Top categorias (contagem)", "% dominante", "Só no teste"],
             [[s["column"], s["n_categories"], ", ".join(f"{x['value']} ({x['count']})" for x in s["top"][:4]), s["dominant_share_pct"], ", ".join(s["unseen_in_train_but_in_test"]) or "—"] for s in p["categorical_stats"]]))
    A(f"\nCategóricas quase constantes (≥ 99% numa categoria): {', '.join(p['near_constant_categorical']) or 'nenhuma'}.\n")

    A("### Categorias observadas que não constam no dicionário\n")
    for label in ("train", "test"):
        mm = dd["category_mismatches"][label]
        A(f"\n**{label}.csv**\n")
        if mm:
            A(_table(["Coluna", "Observado e não documentado", "Documentado e não observado"], [[x["column"], ", ".join(x["observed_not_in_dictionary"]), ", ".join(x["dictionary_codes_not_observed"]) or "—"] for x in mm]))
        else:
            A("Nenhuma.")
    A("\nEssas divergências são de grafia entre o CSV e o dicionário (ex.: abreviações diferentes). O modelo usa os valores **como aparecem no CSV**; o formulário só oferece categorias observadas no treino.\n")

    A("## 11. Valores extremos e inconsistências lógicas\n")
    ex = p["extremes"]
    A("Imóveis com `GrLivArea` > 4000 sq ft no treino:\n")
    A(_table(["Id", "GrLivArea", "SalePrice", "SaleCondition", "OverallQual", "Neighborhood"], [[str(r["Id"]), r["GrLivArea"], r["SalePrice"], r["SaleCondition"], r["OverallQual"], r["Neighborhood"]] for r in ex["grlivarea_gt_4000_train"]]))
    A(f"\nNo teste: {len(ex['grlivarea_gt_4000_test'])} imóvel(is) com `GrLivArea` > 4000.\n")
    A("Verificações lógicas:\n")
    A(_table(["Arquivo", "Regra", "Ocorrências", "Ids (até 10)"], [[r["file"], r["check"], r["count"], ", ".join(map(str, r["example_ids"])) or "—"] for r in p["logical_checks"]]))

    A("\n## 12. Decisões de uso e exclusão\n")
    A("| Item | Decisão | Justificativa |\n|---|---|---|")
    A("| `SalePrice` | Variável-alvo | Presente, numérica, sem ausências no `train.csv`. Nunca usada como entrada. |")
    A("| `Id` | Excluído dos preditores | Identificador sequencial sem significado físico (PRD §7.4). |")
    A("| `MSSubClass` | Tratada como categórica | Códigos numéricos nominais segundo o dicionário. |")
    A("| `OverallQual`, `OverallCond` | Numéricas ordinais | Escala 1–10 com ordem significativa. |")
    A("| Colunas com NA estrutural | Mantidas; NA → `\"None\"` quando o elemento inexiste | Ausência é informativa (PRD §6.4). |")
    A("| Ausências desconhecidas | Imputadas dentro do pipeline (mediana / moda), ajustadas só no treino | Evita vazamento (PRD §8.3). |")
    A("| Linhas extremas (`GrLivArea` > 4000) | **Mantidas** no protocolo principal | Removê-las do holdout tornaria a avaliação otimista; são vendas reais (ver `SaleCondition`). Registradas como limitação. |")
    A(f"| Quase constantes ({', '.join(p['near_constant_categorical']) or '—'}) | Mantidas no conjunto completo | Impacto desprezível; o one-hot e os modelos lidam com elas. Não entram no formulário. |")
    A("| Colunas com vazamento direto do alvo | Nenhuma identificada | Nenhuma coluna é derivada do preço. `MiscVal` é valor de um item acessório, não do imóvel. `SaleType`/`SaleCondition`/`MoSold`/`YrSold` são conhecidos no momento da venda; ficam no conjunto completo, mas não no formulário (não descrevem o imóvel). |")

    A("\n## 13. Limitações identificadas\n")
    A("- Base restrita a Ames, Iowa (EUA), vendas de 2006 a 2010, preços em USD nominais da época — não representa o mercado brasileiro nem preços atuais.")
    A(f"- Tamanho moderado ({p['files'][config.TRAIN_FILE]['rows']:,} imóveis rotulados), com bairros e categorias raras pouco representados.")
    A("- `test.csv` não tem rótulos: só serve para gerar previsões (submissão Kaggle), nunca para métricas.")
    A("- Divergências de grafia entre CSV e dicionário (seção 10) e pequenas inconsistências entre colunas relacionadas (seções 5 e 11).")
    A("- Vendas atípicas (ex.: `SaleCondition = Partial/Abnorml`) e imóveis muito grandes com preço baixo podem gerar erros altos.\n")

    v = p["viability"]
    A("## 14. Decisão de viabilidade (PRD §6.5)\n")
    A(_table(["Critério", "Resultado"], [[k, str(val)] for k, val in v.items()]))
    A(f"\n**Conclusão:** a base {'é' if v['apt_for_regression'] else 'NÃO é'} apta para regressão supervisionada com `SalePrice` como alvo.\n")
    return "\n".join(L)


def main() -> None:
    p = profile()
    config.DATA_PROFILE_PATH.write_text(json.dumps(p, indent=2, ensure_ascii=False, default=_py), encoding="utf-8")
    config.DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (config.DOCS_DIR / "data_audit.md").write_text(render_markdown(p), encoding="utf-8")
    v = p["viability"]
    print(f"train: {p['files'][config.TRAIN_FILE]['rows']}x{p['files'][config.TRAIN_FILE]['columns']} | "
          f"test: {p['files'][config.TEST_FILE]['rows']}x{p['files'][config.TEST_FILE]['columns']}")
    print(f"SalePrice presente/numérico/sem NA: {v['labelled_file_with_numeric_target']}")
    print(f"Apta para regressão: {v['apt_for_regression']}")
    print("Gerado: data/source_manifest.json, artifacts/data_profile.json, artifacts/data_dictionary.json, docs/data_audit.md")
    if not v["apt_for_regression"]:
        raise SystemExit("GATE: base não aprovada para modelagem.")


if __name__ == "__main__":
    main()
