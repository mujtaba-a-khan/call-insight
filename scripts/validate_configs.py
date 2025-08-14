# scripts/validate_configs.py
# Quick sanity check for configs/*.yml
# - Verifies presence and basic types/ranges
# - Ensures tie-break lists reference valid types
# - Checks outcome_window_frac is (0,1]
# - Optionally prints a compact "effective" view
#
# Usage:
#   callinsights-validate
#   callinsights-validate --print-effective
#
# Exit codes:
#   0 = OK
#   1 = Validation error(s)

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CFG_DIR = PROJECT_ROOT / "configs"


def _load_yaml(path: Path) -> Dict:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            if not isinstance(data, dict):
                raise ValueError("YAML root must be a mapping")
            return data
    except FileNotFoundError:
        return {}
    except Exception as e:
        return {"__error__": f"Failed to parse {path.name}: {e}"}


def _validate_app(cfg: Dict) -> List[str]:
    errs: List[str] = []
    if not cfg:
        errs.append("app.yml is missing or empty.")
        return errs

    tz = cfg.get("timezone")
    if not isinstance(tz, str) or not tz:
        errs.append("app.yml: 'timezone' must be a non-empty string.")

    dur = cfg.get("duration_cutoff_s")
    if not isinstance(dur, int) or not (5 <= dur <= 300):
        errs.append("app.yml: 'duration_cutoff_s' must be an int between 5 and 300.")

    if "ingestion" in cfg:
        acc = (cfg["ingestion"] or {}).get("accepted_audio_ext", [])
        if not isinstance(acc, list) or not all(isinstance(x, str) for x in acc):
            errs.append("app.yml: ingestion.accepted_audio_ext must be a list of strings.")

    return errs


def _validate_rules(cfg: Dict) -> List[str]:
    errs: List[str] = []
    if not cfg:
        errs.append("rules.yml is missing or empty.")
        return errs

    conn = cfg.get("connected", {})
    if not isinstance(conn, dict):
        errs.append("rules.yml: 'connected' must be a mapping.")
    else:
        dur = conn.get("duration_cutoff_s")
        tokens = conn.get("min_tokens")
        if not isinstance(dur, int) or dur <= 0:
            errs.append("rules.yml: connected.duration_cutoff_s must be a positive integer.")
        if not isinstance(tokens, int) or tokens <= 0:
            errs.append("rules.yml: connected.min_tokens must be a positive integer.")

    types = cfg.get("types", {})
    if not isinstance(types, dict) or not types:
        errs.append("rules.yml: 'types' must be a non-empty mapping.")
        type_keys = []
    else:
        type_keys = list(types.keys())
        for k, kws in types.items():
            if not isinstance(kws, list) or not all(isinstance(x, str) for x in kws):
                errs.append(f"rules.yml: types.{k} must be a list of strings.")

    tie = cfg.get("type_tie_break", [])
    if not isinstance(tie, list) or not all(isinstance(x, str) for x in tie):
        errs.append("rules.yml: type_tie_break must be a list of strings.")
    else:
        missing = [x for x in tie if x not in type_keys]
        if missing:
            errs.append(f"rules.yml: type_tie_break includes unknown types: {missing}")

    outs = cfg.get("outcomes", {})
    if not isinstance(outs, dict) or not outs:
        errs.append("rules.yml: 'outcomes' must be a non-empty mapping.")
    else:
        for label, phrases in outs.items():
            if label == "negations":
                if not isinstance(phrases, list) or not all(isinstance(x, str) for x in phrases):
                    errs.append("rules.yml: outcomes.negations must be a list of strings.")
                continue
            if not isinstance(phrases, list) or not all(isinstance(x, str) for x in phrases):
                errs.append(f"rules.yml: outcomes.{label} must be a list of strings.")

    frac = cfg.get("outcome_window_frac", 0.25)
    try:
        f = float(frac)
        if not (0.0 < f <= 1.0):
            errs.append("rules.yml: outcome_window_frac must be in (0, 1].")
    except Exception:
        errs.append("rules.yml: outcome_window_frac must be a number in (0, 1].")

    return errs


def _validate_llm(cfg: Dict) -> List[str]:
    errs: List[str] = []
    if not cfg:
        errs.append("llm.yml is missing or empty.")
        return errs

    model = cfg.get("model", {})
    if not isinstance(model, dict):
        errs.append("llm.yml: 'model' must be a mapping.")
    else:
        runner = model.get("runner")
        name = model.get("name")
        if runner not in {"ollama", "none", None}:
            errs.append("llm.yml: model.runner must be 'ollama' or 'none'.")
        if not isinstance(name, str) or not name:
            errs.append("llm.yml: model.name must be a non-empty string.")

    syn = cfg.get("synonyms", {})
    if syn and not isinstance(syn, dict):
        errs.append("llm.yml: 'synonyms' must be a mapping of phrase -> {type_any/outcome_any: [..]}.")

    fs = ((cfg.get("filter_schema") or {}).get("allowed_keys")) if cfg.get("filter_schema") else None
    if fs is not None:
        allowed = {"date_from", "date_to", "type_any", "outcome_any", "agents", "campaigns", "products", "min_amount", "max_amount"}
        if not isinstance(fs, list) or not all(isinstance(x, str) for x in fs):
            errs.append("llm.yml: filter_schema.allowed_keys must be a list of strings.")
        else:
            unknown = [k for k in fs if k not in allowed]
            if unknown:
                errs.append(f"llm.yml: filter_schema.allowed_keys contains unknown keys: {unknown}")

    tz = cfg.get("timezone")
    if tz is not None and not isinstance(tz, str):
        errs.append("llm.yml: 'timezone' must be a string if provided.")

    return errs


def _print_effective(app_cfg: Dict, rules_cfg: Dict, llm_cfg: Dict) -> None:
    """Minimal, readable echo of key effective settings."""
    from llm.synonyms import synonyms_from_config

    syn = synonyms_from_config(llm_cfg)
    print("\n--- Effective Config (compact) ---")
    print(f"timezone: {app_cfg.get('timezone', 'Europe/Berlin')}")
    print(f"duration_cutoff_s: {rules_cfg.get('connected', {}).get('duration_cutoff_s')}")
    print(f"min_tokens: {rules_cfg.get('connected', {}).get('min_tokens')}")
    print(f"types: {', '.join(rules_cfg.get('types', {}).keys())}")
    print(f"type_tie_break: {rules_cfg.get('type_tie_break')}")
    print(f"outcome_window_frac: {rules_cfg.get('outcome_window_frac')}")
    print(f"llm.runner: {llm_cfg.get('model', {}).get('runner')}")
    print(f"llm.name: {llm_cfg.get('model', {}).get('name')}")
    print(f"synonyms: {list(syn.keys())}")
    print("----------------------------------\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate Call Insights YAML configs.")
    parser.add_argument("--print-effective", action="store_true", help="Print a compact 'effective' view after validation.")
    args = parser.parse_args(argv)

    app_cfg = _load_yaml(CFG_DIR / "app.yml")
    rules_cfg = _load_yaml(CFG_DIR / "rules.yml")
    llm_cfg = _load_yaml(CFG_DIR / "llm.yml")

    errors = []
    errors += _validate_app(app_cfg)
    errors += _validate_rules(rules_cfg)
    errors += _validate_llm(llm_cfg)

    if errors:
        print("Config validation failed:\n")
        for e in errors:
            print(f"  • {e}")
        return 1

    print("Configs look good ✅")
    if args.print_effective:
        _print_effective(app_cfg, rules_cfg, llm_cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
