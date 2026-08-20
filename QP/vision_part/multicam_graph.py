import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def parse_participating(s):
    if pd.isna(s) or s == "":
        return []
    return [c.strip() for c in str(s).split(",") if c.strip()]

def camera_participation_counts(df):
    counts = {"cam1": 0, "cam2": 0, "cam3": 0}
    for s in df["participating_cameras"].fillna(""):
        for cam in parse_participating(s):
            if cam in counts:
                counts[cam] += 1
    return counts

def load_and_prepare(csv_path):
    df = pd.read_csv(csv_path)
    # ensure columns exist
    for col in ("overall_x_mean","overall_y_mean","overall_x_std","overall_y_std"):
        if col not in df.columns:
            raise RuntimeError(f"Missing column: {col}")
    x = df["overall_x_mean"].to_numpy(dtype=float)
    y = df["overall_y_mean"].to_numpy(dtype=float)
    xs = df["overall_x_std"].to_numpy(dtype=float)
    ys = df["overall_y_std"].to_numpy(dtype=float)
    valid = (~np.isnan(x)) & (~np.isnan(y))
    return df, x, y, xs, ys, valid

def make_plots(df, x, y, xs, ys, valid, outpath):
    # main single-figure quick save
    plt.figure(figsize=(8,8))
    xi, yi = x[valid], y[valid]
    xsig = np.nan_to_num(xs[valid], nan=0.0)
    ysig = np.nan_to_num(ys[valid], nan=0.0)
    plt.errorbar(xi, yi, xerr=xsig, yerr=ysig, fmt='o-', linewidth=1.5, markersize=5, capsize=3)
    plt.scatter(xi, yi, c=range(len(xi)), cmap='viridis', s=40)
    if xi.size:
        plt.scatter(xi[0], yi[0], c='green', s=120, marker='o', label='Start')
        plt.scatter(xi[-1], yi[-1], c='red', s=120, marker='x', label='End')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title('Robot Trajectory (from CSV)')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath.replace(".png","_main.png"), dpi=300)
    plt.close()

    # full multi-panel
    fig = plt.figure(figsize=(16,12))
    # 1
    ax1 = fig.add_subplot(2,3,1)
    ax1.errorbar(xi, yi, xerr=xsig, yerr=ysig, fmt='o-', linewidth=1.5, markersize=5, capsize=3)
    ax1.scatter(xi, yi, c=range(len(xi)), cmap='viridis', s=40)
    ax1.set_xlabel('X (m)'); ax1.set_ylabel('Y (m)')
    ax1.set_title('Trajectory with Uncertainty'); ax1.grid(True); ax1.axis('equal')

    # 2 X over index
    ax2 = fig.add_subplot(2,3,2)
    idx = np.arange(len(x))
    ax2.errorbar(idx, x, yerr=xs, fmt='r-o', markersize=4, capsize=3)
    ax2.set_xlabel('Position Index'); ax2.set_ylabel('X (m)'); ax2.grid(True)

    # 3 Y over index
    ax3 = fig.add_subplot(2,3,3)
    ax3.errorbar(idx, y, yerr=ys, fmt='g-o', markersize=4, capsize=3)
    ax3.set_xlabel('Position Index'); ax3.set_ylabel('Y (m)'); ax3.grid(True)

    # 4 camera participation
    ax4 = fig.add_subplot(2,3,4)
    counts = camera_participation_counts(df)
    cams = list(counts.keys()); vals = [counts[c] for c in cams]
    ax4.bar(cams, vals, color=['tab:red','tab:green','tab:blue'], alpha=0.8)
    ax4.set_xlabel('Camera'); ax4.set_ylabel('Participation Count'); ax4.set_title('Camera Participation'); ax4.grid(True)

    # 5 uncertainty
    ax5 = fig.add_subplot(2,3,5)
    pos_err = np.sqrt(np.nan_to_num(xs, nan=0.0)**2 + np.nan_to_num(ys, nan=0.0)**2)
    ax5.plot(idx, pos_err, 'purple', marker='o')
    ax5.set_xlabel('Position Index'); ax5.set_ylabel('Position Uncertainty (m)'); ax5.grid(True)

    # 6 stats text
    ax6 = fig.add_subplot(2,3,6); ax6.axis('off')
    total_positions = len(df)
    avg_unc = float(np.nanmean(pos_err)) if pos_err.size else 0.0
    max_unc = float(np.nanmax(pos_err)) if pos_err.size else 0.0
    try:
        x_min = float(np.nanmin(x)); x_max = float(np.nanmax(x))
        y_min = float(np.nanmin(y)); y_max = float(np.nanmax(y))
        last_idx = np.where(valid)[0][-1] if np.any(valid) else -1
        final_x = float(x[last_idx]) if last_idx>=0 else np.nan
        final_y = float(y[last_idx]) if last_idx>=0 else np.nan
    except Exception:
        x_min = x_max = y_min = y_max = final_x = final_y = np.nan

    cam_stats = camera_participation_counts(df)
    stats_text = (
        f"Total positions: {total_positions}\n"
        f"Avg uncertainty: {avg_unc:.6f} m\n"
        f"Max uncertainty: {max_unc:.6f} m\n\n"
        f"Participation: cam1={cam_stats['cam1']}, cam2={cam_stats['cam2']}, cam3={cam_stats['cam3']}\n\n"
        f"X range: [{x_min:.3f}, {x_max:.3f}] m\n"
        f"Y range: [{y_min:.3f}, {y_max:.3f}] m\n\n"
        f"Final valid pos: X={final_x:.4f}, Y={final_y:.4f}"
    )
    ax6.text(0.01, 0.98, stats_text, transform=ax6.transAxes, fontsize=10, va='top', fontfamily='monospace',
             bbox=dict(facecolor='lightgray', alpha=0.6, boxstyle='round,pad=0.5'))

    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv", nargs="?", default="robot_trajectory_auto_20251205_161854.csv", help="input CSV path")
    p.add_argument("-o", "--out", default=None, help="output PNG path")
    args = p.parse_args()

    csv_path = args.csv
    if not os.path.isabs(csv_path):
        csv_path = os.path.join(os.path.dirname(__file__), csv_path)
    if not os.path.exists(csv_path):
        raise SystemExit(f"CSV not found: {csv_path}")

    df, x, y, xs, ys, valid = load_and_prepare(csv_path)
    outpath = args.out or os.path.splitext(csv_path)[0] + "_plot.png"
    make_plots(df, x, y, xs, ys, valid, outpath)
    print(f"Saved: {outpath}")

if __name__ == "__main__":
    main()