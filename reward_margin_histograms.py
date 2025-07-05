import json
import pandas as pd
import matplotlib.pyplot as plt
import argparse

def main():
    parser = argparse.ArgumentParser(description="Plot implicit reward margins for a given step or the last step.")
    parser.add_argument("--jsonl_path", type=str, default="eval_margins.jsonl")
    parser.add_argument("--step", type=int, default=None)
    args = parser.parse_args()

    records = []
    with open(args.jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    if not records:
        raise ValueError(f"No records found in {args.jsonl_path}")

    if args.step is None:
        record = records[-1]
    else:
        matching = [r for r in records if r.get("step") == args.step]
        if not matching:
            raise ValueError(f"No record found for step {args.step}")
        record = matching[0]

    margins = record.get("implicit_margins", [])
    if not margins:
        raise ValueError(f"No implicit_margins found in record for step {record.get('step')}")

    series = pd.Series(margins, name="implicit_margin")
    stats = series.describe().to_frame().T

    print(f"Summary of Implicit Reward Margins (step {record.get('step')})", stats)

    plt.figure()
    plt.hist(margins, bins=30)
    plt.xlabel("Implicit Reward Margin")
    plt.ylabel("Frequency")
    plt.title(f"Histogram of Implicit Reward Margins (step {record.get('step')})")
    plt.show()

if __name__ == "__main__":
    main()
