"""Command-line batch prediction with a trained model.

Usage:
  python predict.py --input transactions.csv --model xgb --output scores.csv
  python predict.py --demo            # score a few labelled transactions from the sample file

The CSV must contain the 165 feature columns local_1..local_93 and agg_1..agg_72
(an optional txId column is carried through).
"""
import argparse
import sys

import pandas as pd

from src.config import PROC_DIR
from src.inference import DISCLAIMER, InputValidationError, predict_frame


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", help="CSV with feature columns")
    ap.add_argument("--model", default="xgb", choices=["logreg", "rf", "xgb"])
    ap.add_argument("--output", default="predictions.csv")
    ap.add_argument("--threshold", type=float, default=None, help="override the validation-tuned threshold")
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args(argv)

    if a.demo:
        df = pd.read_parquet(PROC_DIR / "app_sample.parquet").head(10)
    elif a.input:
        try:
            df = pd.read_csv(a.input)
        except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as e:
            print(f"Input error: cannot read '{a.input}': {e}", file=sys.stderr)
            return 2
    else:
        ap.error("provide --input or --demo")
    try:
        out = predict_frame(df, a.model, a.threshold)
    except InputValidationError as e:
        print(f"Input error: {e}", file=sys.stderr)
        return 2
    out.to_csv(a.output, index=False)
    print(out.head(10).to_string(index=False))
    print(f"\nWrote {len(out)} predictions to {a.output}\n{DISCLAIMER}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
