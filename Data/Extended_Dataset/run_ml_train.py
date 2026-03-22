#!/usr/bin/env python3
"""
merge_attack_only_labels.py

Merge all CSVs under a directory into ONE CSV with ONLY:
  Timestamp + 10 telemetry features + Label

Label rules:
- Label = Attack-type only (no intensity suffix)
- NORMAL => "NORMAL"
- Bonesi samples => "Bonesi-<Attack>" (keeps generator distinction)

Skips:
- results/ folder (default)
- merged outputs (default prefixes: merged_, merged-, merged.)

Fixes:
- broken headers where columns are accidentally named like: 'Smurf','LOW'
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Optional, Tuple, List

import pandas as pd


FEATURE_COLS = [
    "Timestamp",
    "CPU-usage",
    "Num-processes",
    "Interrupts-per-sec",
    "DSK-write",
    "DSK-read",
    "RAM-percentage",
    "Unique-IPs",
    "Num-sockets",
    "Upload-speed",
    "Download-speed",
]

COLUMN_RENAMES = {
    "attacktype": "Attack-type",
    "attack_type": "Attack-type",
    "attack type": "Attack-type",
    "attack-type": "Attack-type",
    "timestamp": "Timestamp",
    "intesnity": "Intensity",
    "intensity": "Intensity",
}

ATTACK_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\bnormal\b|\bbenign\b", re.I), "NORMAL"),
    (re.compile(r"\budp[\s_-]*flood\b", re.I), "UDP Flood"),
    (re.compile(r"\btcp[\s_-]*flood\b", re.I), "TCP Flood"),
    (re.compile(r"\bsyn[\s_-]*flood\b", re.I), "SYN Flood"),
    (re.compile(r"\bfin[\s_-]*flood\b", re.I), "FIN Flood"),
    (re.compile(r"\breset[\s_-]*flood\b", re.I), "Reset Flood"),
    (re.compile(r"\bpush[\s_-]*ack\b", re.I), "PUSH ACK"),
    (re.compile(r"\bsyn[\s_-]*fin\b", re.I), "SYN FIN"),
    (re.compile(r"\bsmurf\b", re.I), "Smurf"),
    (re.compile(r"\bslowloris\b", re.I), "SlowLoris"),
    (re.compile(r"\bslowread\b", re.I), "SlowRead"),
    (re.compile(r"\brudy\b", re.I), "RUDY"),
    (re.compile(r"\bhttp[\s_-]*flood\b", re.I), "HTTP Flood"),
    (re.compile(r"\bicmp[\s_-]*flood\b", re.I), "ICMP Flood"),
    (re.compile(r"\bdirbuster\b", re.I), "DirBuster"),
    (re.compile(r"\bsql[\s_-]*injection\b|\bsqli\b", re.I), "SQL injection"),
]


def sniff_delimiter(path: Path) -> str:
    try:
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
            sample = f.read(8192)
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"])
        return dialect.delimiter
    except Exception:
        return ","


def canon_colname(name: str) -> str:
    raw = str(name).strip()
    key = raw.lower().replace("-", " ").replace("_", " ")
    key = re.sub(r"\s+", " ", key).strip()
    compact = re.sub(r"[^a-z0-9]+", "", key)
    if compact in COLUMN_RENAMES:
        return COLUMN_RENAMES[compact]
    if key in COLUMN_RENAMES:
        return COLUMN_RENAMES[key]
    return raw


def normalize_attack(text: str) -> Optional[str]:
    for pat, name in ATTACK_PATTERNS:
        if pat.search(text):
            return name
    return None


def fix_broken_header_attack_intensity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix case where columns are literally named like:
      ... , Smurf , LOW
    instead of:
      ... , Attack-type , Intensity
    We only really need Attack-type, but this rename helps inference.
    """
    cols = list(df.columns)
    if "Attack-type" in cols:
        return df

    attack_cols = [c for c in cols if normalize_attack(str(c)) is not None]
    if len(attack_cols) == 1:
        return df.rename(columns={attack_cols[0]: "Attack-type"})
    return df


def read_csv_any(path: Path) -> pd.DataFrame:
    sep = sniff_delimiter(path)
    df = pd.read_csv(path, sep=sep, engine="python")
    df.columns = [canon_colname(c) for c in df.columns]
    df = fix_broken_header_attack_intensity(df)
    return df


def infer_attack_only(df: pd.DataFrame, file_path: Path) -> Tuple[str, bool]:
    """
    Returns: (attack, is_bonesi)
    """
    def clean_str(x) -> str:
        return str(x).strip()

    # tokens from full path
    full_text = " ".join([str(p) for p in file_path.parts]).lower()
    tokens = re.split(r"[\/\\\s_\-\.]+", full_text)
    tokens = [t for t in tokens if t]
    is_bonesi = any("bonesi" in t for t in tokens)

    attack = None
    if "Attack-type" in df.columns:
        s = df["Attack-type"].astype(str).map(clean_str)
        s = s[s.notna() & (s != "") & (s.str.lower() != "nan")]
        if len(s) > 0:
            attack = clean_str(s.mode().iloc[0])

    if not attack:
        hint = " ".join(file_path.parts)
        attack = normalize_attack(hint)

    if not attack:
        attack = "UNKNOWN"
    else:
        attack = normalize_attack(attack) or attack

    return attack, is_bonesi


def make_label(attack: str, is_bonesi: bool) -> str:
    attack = (attack or "UNKNOWN").strip()
    if attack.upper() == "NORMAL":
        return "NORMAL"
    base = attack.replace(" ", "-")
    return f"Bonesi-{base}" if is_bonesi else base


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Root folder to scan recursively for CSV files.")
    ap.add_argument("--output", required=True, help="Output merged CSV.")
    ap.add_argument("--keep-unknown", action="store_true", help="Keep UNKNOWN-labeled files.")
    ap.add_argument("--exclude-dirs", default="results,.git,__pycache__", help="Comma-separated dir names to skip.")
    ap.add_argument("--exclude-prefixes", default="merged_,merged-,merged.", help="Comma-separated filename prefixes to skip.")
    args = ap.parse_args()

    root = Path(args.input).expanduser().resolve()
    out = Path(args.output).expanduser().resolve()

    exclude_dirs = set([x.strip() for x in args.exclude_dirs.split(",") if x.strip()])
    exclude_prefixes = tuple([x.strip().lower() for x in args.exclude_prefixes.split(",") if x.strip()])

    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Input directory not found: {root}")

    csv_files = []
    for p in root.rglob("*"):
        if not p.is_file() or p.suffix.lower() != ".csv":
            continue
        if any(part in exclude_dirs for part in p.parts):
            continue
        if p.name.lower().startswith(exclude_prefixes):
            continue
        csv_files.append(p)

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under: {root}")

    frames = []
    for p in sorted(csv_files):
        try:
            df = read_csv_any(p)
        except Exception as e:
            print(f"[WARN] Skipping unreadable file: {p} ({e})")
            continue

        attack, is_bonesi = infer_attack_only(df, p)

        if attack == "UNKNOWN" and not args.keep_unknown:
            print(f"[WARN] Dropping UNKNOWN-labeled file: {p}")
            continue

        df["Label"] = make_label(attack, is_bonesi)

        keep = [c for c in FEATURE_COLS if c in df.columns] + ["Label"]
        df = df[keep]
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True, sort=False)

    for c in FEATURE_COLS:
        if c not in merged.columns:
            merged[c] = pd.NA
    merged = merged[FEATURE_COLS + ["Label"]]

    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out, index=False)

    print(f"[OK] Wrote: {out}")
    print(f"[OK] Rows: {len(merged):,} | Cols: {len(merged.columns)}")
    print("[OK] Example labels:", merged["Label"].dropna().unique()[:20])


if __name__ == "__main__":
    main()
