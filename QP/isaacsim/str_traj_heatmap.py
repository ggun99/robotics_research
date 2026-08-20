import os
import re
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

### python3 str_traj_heatmap.py -r /path/to/results_folder
FNAME_RE = re.compile(r"error_w1_([-+]?\d*\.?\d+)_w2_([-+]?\d*\.?\d+)_w4_([-+]?\d*\.?\d+)_lambda_([-+]?\d*\.?\d+)\.txt", re.IGNORECASE)

def load_results(results_folder):
    rows = []
    files = sorted(glob.glob(os.path.join(results_folder, "error_*.txt")))
    print(f"[debug] searching folder: {results_folder} -> {len(files)} files matched glob 'error_*.txt'")
    for p in files:
        bn = os.path.basename(p)
        m = FNAME_RE.search(bn)
        if not m:
            # debug: report files that don't match expected pattern
            print(f"[debug] file skipped (pattern mismatch): {bn}")
            continue
        w1, w2, w4, lam = m.groups()
        try:
            with open(p, "r") as f:
                s = f.read().strip()
            parts = [x.strip() for x in s.split(",") if x.strip() != ""]
            final_error = float(parts[0]) if len(parts) >= 1 else np.nan
            final_error_base = float(parts[1]) if len(parts) >= 2 else np.nan
            # normalize known sentinel value 50 -> 5 (use isclose for safety)
            if not np.isnan(final_error) and np.isclose(final_error, 50.0):
                print(f"[debug] adjusting final_error 50->5 for file {bn}")
                final_error = 4.0
            if not np.isnan(final_error_base) and np.isclose(final_error_base, 50.0):
                print(f"[debug] adjusting final_error_base 50->5 for file {bn}")
                final_error_base = 4.0
        except Exception:
            final_error = np.nan
            final_error_base = np.nan
        rows.append({
            "w1": float(w1), "w2": float(w2), "w4": float(w4), "lambda": float(lam),
            "final_error": final_error, "final_error_base": final_error_base,
            "file": p
        })
    df = pd.DataFrame(rows)
    return df

def ensure_grid(df, w1_vals=None, w2_vals=None, w4_vals=None, lambda_vals=None):
    if w1_vals is None:
        w1_vals = sorted(df["w1"].unique())
    if w2_vals is None:
        w2_vals = sorted(df["w2"].unique())
    if w4_vals is None:
        w4_vals = sorted(df["w4"].unique())
    if lambda_vals is None:
        lambda_vals = sorted(df["lambda"].unique())
    return w1_vals, w2_vals, w4_vals, lambda_vals

def save_heatmap(mat, x_labels, y_labels, title, outpath, vmin=0.0, vmax=4.0, annotate=True):
    plt.figure(figsize=(6,5))
    sns.set(font_scale=1.0)
    ax = sns.heatmap(mat, xticklabels=x_labels, yticklabels=y_labels, annot=annotate, fmt=".3f",
                     cmap="YlGnBu", vmin=vmin, vmax=vmax, cbar_kws={"label":"error"})
    ax.set_xlabel("x = w_m")
    ax.set_ylabel("y = w_h")
    # ax.set_title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def plot_fix_w1(df, results_folder, w_values=None):
    w1_vals, w2_vals, w4_vals, _ = ensure_grid(df, w1_vals=w_values)
    outdir = os.path.join(results_folder, "heatmaps_fix_w1")
    os.makedirs(outdir, exist_ok=True)
    for w1 in w1_vals:
        sub = df[df["w1"] == w1]
        pivot_err = sub.pivot_table(index="w4", columns="w2", values="final_error", aggfunc=np.mean)
        pivot_err_base = sub.pivot_table(index="w4", columns="w2", values="final_error_base", aggfunc=np.mean)
        # ensure ordering
        pivot_err = pivot_err.reindex(index=sorted(pivot_err.index), columns=sorted(pivot_err.columns))
        pivot_err_base = pivot_err_base.reindex(index=sorted(pivot_err_base.index), columns=sorted(pivot_err_base.columns))
        save_heatmap(pivot_err.values, pivot_err.columns, pivot_err.index,
                     f"error_EE (w_m={w1})  x=w_o  y=w_h",
                     os.path.join(outdir, f"heat_final_error_w1_{w1}.png"),vmin = 0.2, vmax=0.32)
        save_heatmap(pivot_err_base.values, pivot_err_base.columns, pivot_err_base.index,
                     f"error_Base (w_m={w1})  x=w_o  y=w_h",
                     os.path.join(outdir, f"heat_final_error_base_w1_{w1}.png"),vmin= 1.9, vmax=2.7)

def plot_fix_w2_and_w4_variations(df, results_folder):
    w1_vals, w2_vals, w4_vals, _ = ensure_grid(df)
    outdir = os.path.join(results_folder, "heatmaps_fix_w2_w4")
    os.makedirs(outdir, exist_ok=True)
    # for each fixed w2 -> plot x=w1, y=w4
    for w2 in w2_vals:
        sub = df[df["w2"] == w2]
        pivot = sub.pivot_table(index="w4", columns="w1", values="final_error", aggfunc='mean')
        pivot = pivot.reindex(index=sorted(pivot.index), columns=sorted(pivot.columns))
        save_heatmap(pivot.values, pivot.columns, pivot.index,
                     f"error_EE (w2={w2})  x=w_m  y=w_h",
                     os.path.join(outdir, f"heat_final_error_fix_w2_{w2}.png"),vmin = 0.2, vmax=0.32)
        # also plot final_error_base
        pivot_base = sub.pivot_table(index="w4", columns="w1", values="final_error_base", aggfunc='mean')
        pivot_base = pivot_base.reindex(index=sorted(pivot_base.index), columns=sorted(pivot_base.columns))
        save_heatmap(pivot_base.values, pivot_base.columns, pivot_base.index,
                     f"error_Base (w2={w2})  x=w_m  y=w_h",
                     os.path.join(outdir, f"heat_final_error_base_fix_w2_{w2}.png"),vmin= 1.9, vmax=2.7)
    # for each fixed w4 -> plot x=w1, y=w2
    for w4 in w4_vals:
        sub = df[df["w4"] == w4]
        pivot = sub.pivot_table(index="w2", columns="w1", values="final_error", aggfunc='mean')
        pivot = pivot.reindex(index=sorted(pivot.index), columns=sorted(pivot.columns))
        save_heatmap(pivot.values, pivot.columns, pivot.index,
                     f"error_EE (w4={w4})  x=w_m  y=w_o",
                     os.path.join(outdir, f"heat_final_error_fix_w4_{w4}.png"),vmin = 0.2, vmax=0.32)
        # also plot final_error_base
        pivot_base = sub.pivot_table(index="w2", columns="w1", values="final_error_base", aggfunc='mean')
        pivot_base = pivot_base.reindex(index=sorted(pivot_base.index), columns=sorted(pivot_base.columns))
        save_heatmap(pivot_base.values, pivot_base.columns, pivot_base.index,
                     f"error_Base (w4={w4})  x=w_m  y=w_o",
                     os.path.join(outdir, f"heat_final_error_base_fix_w4_{w4}.png"),vmin= 1.9, vmax=2.7)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-folder", "-r", required=True)
    ap.add_argument("--w-values", nargs="*", type=float, help="optional w-values order to force")
    args = ap.parse_args()

    df = load_results(args.results_folder)
    if df.empty:
        print("No error files found in folder:", args.results_folder)
        return
    # save summary CSV
    df.to_csv(os.path.join(args.results_folder, "errors_summary.csv"), index=False)
    # plot_fix_w1(df, args.results_folder, w_values=args.w_values)
    plot_fix_w2_and_w4_variations(df, args.results_folder)
    print("Heatmaps saved in:", os.path.join(args.results_folder, "heatmaps_fix_w1"))
    # print("Heatmaps saved in:", os.path.join(args.results_folder, "heatmaps_fix_w1"),
    #       "and", os.path.join(args.results_folder, "heatmaps_fix_w2_w4"))

if __name__ == "__main__":
    main()