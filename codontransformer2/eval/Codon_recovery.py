"""Codon usage metrics for CodonTransformer-style evaluation.

Implements:
  1. Formal CSI
     - CAI-style relative adaptiveness weights
     - Reference source: species-wide codon usage table
     - w_ij = f_ij / max_synonymous_frequency_for_amino_acid_i
     - CSI = exp(mean(log(w_k)))

  2. CAI
     - Standard CAI formula
     - Reference source: user-provided reference gene set
       e.g. highly expressed genes, or CodonTransformer-style top-10%-CSI training genes
     - w_ij = reference_count_ij / max_synonymous_reference_count_for_amino_acid_i
     - CAI = exp(mean(log(w_k)))

  3. %MinMax
     - Local codon commonness/rarity profile
     - Per codon:
         +100 = most frequent synonymous codon
            0 = average synonymous usage
         -100 = rarest synonymous codon
     - Then smooth with sliding window, default 18 codons.

Input assumptions:
  - DNA sequences are coding DNA sequences in-frame.
  - Codon usage table has at least:
        species, aa, codon, frequency
  - Reference genes for CAI have at least:
        species, dna
    or equivalent column names can be passed.

This file can be imported as a library or run as a CLI.
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


GENETIC_CODE: Dict[str, str] = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}

SENSE_CODONS: List[str] = [c for c, aa in GENETIC_CODE.items() if aa != "*"]

SYNONYMOUS_CODONS: Dict[str, List[str]] = defaultdict(list)
for _codon, _aa in GENETIC_CODE.items():
    if _aa != "*":
        SYNONYMOUS_CODONS[_aa].append(_codon)


# =============================================================================
# Basic sequence utilities
# =============================================================================

def clean_dna(seq) -> str:
    """Normalize a DNA/RNA sequence string."""
    if seq is None or pd.isna(seq):
        return ""
    return (
        str(seq)
        .upper()
        .replace("U", "T")
        .replace(" ", "")
        .replace("\n", "")
        .replace("\r", "")
        .strip()
    )


def iter_sense_codons(seq: str) -> Iterable[Tuple[str, str]]:
    """
    Yield (aa, codon) for valid sense codons only.
    Ambiguous codons and stops are skipped.
    """
    seq = clean_dna(seq)

    for i in range(0, len(seq) - 2, 3):
        codon = seq[i:i + 3]

        if len(codon) != 3:
            continue
        if any(base not in "ACGT" for base in codon):
            continue

        aa = GENETIC_CODE.get(codon)
        if aa is None or aa == "*":
            continue

        yield aa, codon


def geometric_mean_from_logs(logs: List[float]) -> float:
    if not logs:
        return float("nan")
    return float(math.exp(sum(logs) / len(logs)))


# =============================================================================
# Codon table loading
# =============================================================================

def read_table(path: str | Path) -> pd.DataFrame:
    """Read CSV or Parquet by suffix."""
    path = Path(path)

    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)

    return pd.read_csv(path)


def normalize_codon_frequency_table(
    codon_table: pd.DataFrame,
    species_col: str = "species",
    aa_col: str = "aa",
    codon_col: str = "codon",
    frequency_col: str = "frequency",
) -> pd.DataFrame:
    """
    Return a standardized codon frequency table with columns:
      species, aa, codon, frequency
    """
    needed = [species_col, aa_col, codon_col, frequency_col]
    missing = [c for c in needed if c not in codon_table.columns]
    if missing:
        raise ValueError(f"Codon table missing columns: {missing}")

    out = codon_table[needed].copy()
    out.columns = ["species", "aa", "codon", "frequency"]

    out["species"] = out["species"].astype(str)
    out["aa"] = out["aa"].astype(str)
    out["codon"] = out["codon"].astype(str).str.upper().str.replace("U", "T")
    out["frequency"] = out["frequency"].astype(float)

    out = out[out["codon"].isin(SENSE_CODONS)].copy()
    out = out[out["frequency"] > 0].copy()

    return out


# =============================================================================
# Formal CSI
# =============================================================================

def build_formal_csi_weights(
    codon_freq: pd.DataFrame,
) -> Dict[Tuple[str, str, str], float]:
    """
    Build formal CSI weights from species-wide codon frequencies.

    For each species and amino acid:
      w(species, aa, codon) = frequency(species, aa, codon)
                               / max_frequency(species, aa)

    This is the CodonTransformer-style CSI definition:
      same relative adaptiveness formula as CAI,
      but using species-wide codon usage as reference.
    """
    codon_freq = normalize_codon_frequency_table(codon_freq)

    max_freq = (
        codon_freq
        .groupby(["species", "aa"], as_index=False)["frequency"]
        .max()
        .rename(columns={"frequency": "max_frequency"})
    )

    use = codon_freq.merge(max_freq, on=["species", "aa"], how="left")
    use["weight"] = use["frequency"] / use["max_frequency"]

    weights: Dict[Tuple[str, str, str], float] = {}

    for r in use.itertuples(index=False):
        if r.weight > 0:
            weights[(str(r.species), str(r.aa), str(r.codon))] = float(r.weight)

    return weights


def score_weight_geomean(
    dna: str,
    species: str,
    weights: Dict[Tuple[str, str, str], float],
) -> Tuple[float, int, int]:
    """
    Generic geometric-mean scorer used by CSI and CAI.

    Returns:
      score, total_sense_codons, missing_weight_count
    """
    species = str(species)
    logs: List[float] = []
    total = 0
    missing = 0

    for aa, codon in iter_sense_codons(dna):
        total += 1
        w = weights.get((species, aa, codon))

        if w is None or w <= 0:
            missing += 1
            continue

        logs.append(math.log(w))

    return geometric_mean_from_logs(logs), total, missing


def formal_csi(
    dna: str,
    species: str,
    csi_weights: Dict[Tuple[str, str, str], float],
) -> Tuple[float, int, int]:
    """
    Formal CSI score for one sequence against its species-wide codon table.

    Returns:
      csi, total_sense_codons, missing_weight_count
    """
    return score_weight_geomean(dna, species, csi_weights)


# =============================================================================
# CAI
# =============================================================================

def build_cai_weights_from_reference_genes(
    reference_genes: pd.DataFrame,
    species_col: str = "species",
    dna_col: str = "dna",
    pseudocount: float = 1.0,
) -> Dict[Tuple[str, str, str], float]:
    """
    Build standard CAI weights from a reference gene set.

    The reference can be:
      - experimentally high-expression genes,
      - ribosomal genes,
      - CodonTransformer-style top-10%-CSI training genes.

    For each species and amino acid:
      count_ij = count of codon j in reference genes
      w_ij = count_ij / max_synonymous_count_i

    Pseudocount avoids zero weights when the reference set is small.
    """
    if species_col not in reference_genes.columns:
        raise ValueError(f"reference_genes missing species column: {species_col}")
    if dna_col not in reference_genes.columns:
        raise ValueError(f"reference_genes missing DNA column: {dna_col}")

    ref = reference_genes[[species_col, dna_col]].copy()
    ref.columns = ["species", "dna"]
    ref["species"] = ref["species"].astype(str)

    species_set = sorted(ref["species"].dropna().astype(str).unique().tolist())

    counts: Dict[Tuple[str, str, str], float] = defaultdict(float)

    # Initialize with pseudocount.
    for species in species_set:
        for aa, codons in SYNONYMOUS_CODONS.items():
            for codon in codons:
                counts[(species, aa, codon)] += float(pseudocount)

    # Count reference codons.
    for r in ref.itertuples(index=False):
        species = str(r.species)
        for aa, codon in iter_sense_codons(r.dna):
            counts[(species, aa, codon)] += 1.0

    # Find max count per species/amino acid.
    max_by_species_aa: Dict[Tuple[str, str], float] = defaultdict(float)

    for (species, aa, codon), count in counts.items():
        if count > max_by_species_aa[(species, aa)]:
            max_by_species_aa[(species, aa)] = count

    # Relative adaptiveness weights.
    weights: Dict[Tuple[str, str, str], float] = {}

    for key, count in counts.items():
        species, aa, codon = key
        max_count = max_by_species_aa[(species, aa)]

        if max_count > 0:
            weights[key] = float(count / max_count)

    return weights


def cai(
    dna: str,
    species: str,
    cai_weights: Dict[Tuple[str, str, str], float],
) -> Tuple[float, int, int]:
    """
    Standard CAI score for one sequence against a reference gene set.

    Returns:
      cai, total_sense_codons, missing_weight_count
    """
    return score_weight_geomean(dna, species, cai_weights)


# =============================================================================
# Top-10%-CSI training reference construction
# =============================================================================

def select_top_fraction_by_csi(
    genes: pd.DataFrame,
    codon_freq: pd.DataFrame,
    species_col: str = "species",
    dna_col: str = "dna",
    top_frac: float = 0.10,
) -> pd.DataFrame:
    """
    Select top fraction of genes per species by formal CSI.

    This is useful for CodonTransformer-style CAI:
      1. Score training/reference genes by formal CSI.
      2. Select top 10% per species.
      3. Build CAI weights from those genes.
    """
    if not 0 < top_frac <= 1:
        raise ValueError("top_frac must be in (0, 1].")

    if species_col not in genes.columns:
        raise ValueError(f"genes missing species column: {species_col}")
    if dna_col not in genes.columns:
        raise ValueError(f"genes missing DNA column: {dna_col}")

    genes = genes.copy()
    csi_weights = build_formal_csi_weights(codon_freq)

    scores = []

    for idx, r in genes.iterrows():
        score, total, missing = formal_csi(
            dna=r[dna_col],
            species=str(r[species_col]),
            csi_weights=csi_weights,
        )
        scores.append((idx, score, total, missing))

    score_df = pd.DataFrame(
        scores,
        columns=["_idx", "formal_csi", "codons_total", "codons_missing_weight"],
    ).set_index("_idx")

    genes = genes.join(score_df)

    selected = []

    for species, g in genes.dropna(subset=["formal_csi"]).groupby(species_col, sort=False):
        g = g.sort_values("formal_csi", ascending=False)
        n_top = max(1, int(math.ceil(top_frac * len(g))))
        selected.append(g.head(n_top))

    if not selected:
        return genes.iloc[0:0].copy()

    return pd.concat(selected, ignore_index=True)


# =============================================================================
# %MinMax
# =============================================================================

def build_minmax_tables(
    codon_freq: pd.DataFrame,
) -> Tuple[
    Dict[Tuple[str, str, str], float],
    Dict[Tuple[str, str], Dict[str, float]],
]:
    """
    Build lookup tables for %MinMax.

    For each species/amino acid:
      min = rarest synonymous codon frequency
      max = most common synonymous codon frequency
      mean = average synonymous codon frequency
    """
    codon_freq = normalize_codon_frequency_table(codon_freq)

    freq_lookup: Dict[Tuple[str, str, str], float] = {}

    for r in codon_freq.itertuples(index=False):
        freq_lookup[(str(r.species), str(r.aa), str(r.codon))] = float(r.frequency)

    stats = (
        codon_freq
        .groupby(["species", "aa"])["frequency"]
        .agg(["min", "max", "mean", "count"])
        .reset_index()
    )

    stat_lookup: Dict[Tuple[str, str], Dict[str, float]] = {}

    for r in stats.itertuples(index=False):
        stat_lookup[(str(r.species), str(r.aa))] = {
            "min": float(r.min),
            "max": float(r.max),
            "mean": float(r.mean),
            "count": int(r.count),
        }

    return freq_lookup, stat_lookup


def minmax_codon_score(
    species: str,
    aa: str,
    codon: str,
    freq_lookup: Dict[Tuple[str, str, str], float],
    stat_lookup: Dict[Tuple[str, str], Dict[str, float]],
) -> Optional[float]:
    """
    Per-codon %MinMax score.

    +100 = most frequent synonymous codon
       0 = average synonymous usage
    -100 = rarest synonymous codon
    """
    species = str(species)
    codon = str(codon).upper().replace("U", "T")

    f = freq_lookup.get((species, aa, codon))
    st = stat_lookup.get((species, aa))

    if f is None or st is None:
        return None

    mn = float(st["min"])
    mx = float(st["max"])
    avg = float(st["mean"])

    # Single-codon amino acids or degenerate cases are neutral.
    if int(st["count"]) <= 1 or mx == mn:
        return 0.0

    if f >= avg:
        denom = mx - avg
        if denom <= 0:
            return 0.0
        return float(100.0 * (f - avg) / denom)

    denom = avg - mn
    if denom <= 0:
        return 0.0

    return float(-100.0 * (avg - f) / denom)


def minmax_raw_scores(
    dna: str,
    species: str,
    freq_lookup: Dict[Tuple[str, str, str], float],
    stat_lookup: Dict[Tuple[str, str], Dict[str, float]],
) -> Tuple[np.ndarray, int, int]:
    """
    Return raw per-codon %MinMax scores before sliding-window smoothing.

    Returns:
      raw_scores, total_sense_codons, missing_score_count
    """
    scores: List[float] = []
    total = 0
    missing = 0

    for aa, codon in iter_sense_codons(dna):
        total += 1
        s = minmax_codon_score(species, aa, codon, freq_lookup, stat_lookup)

        if s is None:
            missing += 1
            continue

        scores.append(float(s))

    return np.asarray(scores, dtype=np.float32), total, missing


def rolling_mean(x: np.ndarray, window: int = 18) -> np.ndarray:
    """
    Sliding-window mean.

    If sequence is shorter than window, return a single mean point.
    """
    if len(x) == 0:
        return np.asarray([], dtype=np.float32)

    if len(x) < window:
        return np.asarray([float(np.mean(x))], dtype=np.float32)

    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(x, kernel, mode="valid").astype(np.float32)


def minmax_profile(
    dna: str,
    species: str,
    freq_lookup: Dict[Tuple[str, str, str], float],
    stat_lookup: Dict[Tuple[str, str], Dict[str, float]],
    window: int = 18,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """
    Compute raw and window-smoothed %MinMax profile.

    Returns:
      raw_scores, profile, total_sense_codons, missing_score_count
    """
    raw, total, missing = minmax_raw_scores(dna, species, freq_lookup, stat_lookup)
    profile = rolling_mean(raw, window=window)
    return raw, profile, total, missing


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))

    if n < 2:
        return float("nan")

    a = a[:n]
    b = b[:n]

    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")

    return float(np.corrcoef(a, b)[0, 1])


def compare_minmax_profiles(
    native_dna: str,
    pred_dna: str,
    species: str,
    freq_lookup: Dict[Tuple[str, str, str], float],
    stat_lookup: Dict[Tuple[str, str], Dict[str, float]],
    window: int = 18,
) -> Dict[str, float]:
    """
    Compare predicted %MinMax profile against native profile.

    Returns:
      native_minmax_mean
      pred_minmax_mean
      abs_mean_minmax_diff
      profile_mae
      profile_rmse
      profile_corr
      missing fractions
    """
    native_raw, native_profile, native_total, native_missing = minmax_profile(
        native_dna, species, freq_lookup, stat_lookup, window=window
    )
    pred_raw, pred_profile, pred_total, pred_missing = minmax_profile(
        pred_dna, species, freq_lookup, stat_lookup, window=window
    )

    n = min(len(native_profile), len(pred_profile))

    if n == 0:
        profile_mae = float("nan")
        profile_rmse = float("nan")
        profile_corr = float("nan")
    else:
        diff = pred_profile[:n] - native_profile[:n]
        profile_mae = float(np.mean(np.abs(diff)))
        profile_rmse = float(np.sqrt(np.mean(diff ** 2)))
        profile_corr = safe_corr(native_profile, pred_profile)

    native_mean = float(np.mean(native_raw)) if len(native_raw) else float("nan")
    pred_mean = float(np.mean(pred_raw)) if len(pred_raw) else float("nan")

    if math.isnan(native_mean) or math.isnan(pred_mean):
        abs_mean_diff = float("nan")
    else:
        abs_mean_diff = float(abs(pred_mean - native_mean))

    return {
        "native_codons": native_total,
        "pred_codons": pred_total,
        "native_missing_frac": native_missing / max(native_total, 1),
        "pred_missing_frac": pred_missing / max(pred_total, 1),
        "native_minmax_mean": native_mean,
        "pred_minmax_mean": pred_mean,
        "abs_mean_minmax_diff": abs_mean_diff,
        "profile_points_compared": n,
        "profile_mae": profile_mae,
        "profile_rmse": profile_rmse,
        "profile_corr": profile_corr,
    }


# =============================================================================
# CLI
# =============================================================================

def cli_score_predictions(args: argparse.Namespace) -> None:
    """
    Score a prediction CSV with CSI, optional CAI, and optional MinMax.

    Required prediction columns:
      model, species, pred_dna

    Required for MinMax comparison:
      native_dna
    """
    pred = read_table(args.predictions)
    codon_freq = read_table(args.codon_table)

    if args.model_col not in pred.columns:
        raise ValueError(f"Predictions missing model column: {args.model_col}")
    if args.species_col not in pred.columns:
        raise ValueError(f"Predictions missing species column: {args.species_col}")
    if args.pred_dna_col not in pred.columns:
        raise ValueError(f"Predictions missing predicted DNA column: {args.pred_dna_col}")

    pred = pred.copy()
    pred = pred[pred[args.pred_dna_col].fillna("").astype(str).str.len() > 0].copy()

    csi_weights = build_formal_csi_weights(codon_freq)

    cai_weights = None
    if args.cai_reference is not None:
        ref = read_table(args.cai_reference)
        cai_weights = build_cai_weights_from_reference_genes(
            reference_genes=ref,
            species_col=args.cai_species_col,
            dna_col=args.cai_dna_col,
            pseudocount=args.pseudocount,
        )

    minmax_freq = None
    minmax_stats = None
    if args.do_minmax:
        if args.native_dna_col not in pred.columns:
            raise ValueError(
                f"--do_minmax requires native DNA column: {args.native_dna_col}"
            )
        minmax_freq, minmax_stats = build_minmax_tables(codon_freq)

    rows = []

    for idx, r in pred.iterrows():
        model = str(r[args.model_col])
        species = str(r[args.species_col])
        pred_dna = r[args.pred_dna_col]

        csi_score, csi_total, csi_missing = formal_csi(
            pred_dna,
            species,
            csi_weights,
        )

        row = {
            "row_id": idx,
            "model": model,
            "species": species,
            "formal_csi": csi_score,
            "formal_csi_codons_total": csi_total,
            "formal_csi_missing_weight_frac": csi_missing / max(csi_total, 1),
        }

        if cai_weights is not None:
            cai_score_value, cai_total, cai_missing = cai(
                pred_dna,
                species,
                cai_weights,
            )
            row.update({
                "cai": cai_score_value,
                "cai_codons_total": cai_total,
                "cai_missing_weight_frac": cai_missing / max(cai_total, 1),
            })

        if args.do_minmax:
            mm = compare_minmax_profiles(
                native_dna=r[args.native_dna_col],
                pred_dna=pred_dna,
                species=species,
                freq_lookup=minmax_freq,
                stat_lookup=minmax_stats,
                window=args.window,
            )
            row.update(mm)

        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(f"{args.out_prefix}_by_sequence.csv", index=False)

    agg = {
        "formal_csi": "mean",
        "formal_csi_missing_weight_frac": "mean",
    }

    if cai_weights is not None:
        agg["cai"] = "mean"
        agg["cai_missing_weight_frac"] = "mean"

    if args.do_minmax:
        agg.update({
            "abs_mean_minmax_diff": "mean",
            "profile_mae": "mean",
            "profile_rmse": "mean",
            "profile_corr": "mean",
            "pred_missing_frac": "mean",
        })

    summary = (
        out.groupby("model", as_index=False)
        .agg(n=("row_id", "count"), **{k: (k, v) for k, v in agg.items()})
    )

    summary.to_csv(f"{args.out_prefix}_summary_by_model.csv", index=False)

    print("\n=== Summary by model ===")
    print(summary.to_string(index=False))
    print(f"\nWROTE {args.out_prefix}_by_sequence.csv")
    print(f"WROTE {args.out_prefix}_summary_by_model.csv")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compute formal CSI, standard CAI, and %MinMax for codon sequences."
    )

    ap.add_argument("--predictions", required=True)
    ap.add_argument("--codon_table", required=True)
    ap.add_argument("--out_prefix", required=True)

    ap.add_argument("--model_col", default="model")
    ap.add_argument("--species_col", default="species")
    ap.add_argument("--pred_dna_col", default="pred_dna")
    ap.add_argument("--native_dna_col", default="native_dna")

    ap.add_argument(
        "--cai_reference",
        default=None,
        help=(
            "Optional reference genes table for CAI. "
            "Must contain species and DNA columns. "
            "Use high-expression genes or top-10%-CSI training genes."
        ),
    )
    ap.add_argument("--cai_species_col", default="species")
    ap.add_argument("--cai_dna_col", default="dna")
    ap.add_argument("--pseudocount", type=float, default=1.0)

    ap.add_argument("--do_minmax", action="store_true")
    ap.add_argument("--window", type=int, default=18)

    args = ap.parse_args()
    cli_score_predictions(args)


if __name__ == "__main__":
    main()
