"""
Congestion and Traffic Density Temporal Plot
=============================================

Reads the per-frame metrics CSV saved by main.py and produces a
multi-panel plot suitable for the report and demo.

Usage
-----
    # First generate metrics CSV from a video:
    python main.py your_video.mp4 --output-metrics output/metrics.csv

    # Then plot:
    python plot_analytics.py output/metrics.csv

    # Or specify output path:
    python plot_analytics.py output/metrics.csv --output output/congestion_plot.png
"""

import os
import sys
import argparse

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def plot_analytics(csv_path: str, output_path: str = None, show: bool = True):
    df = pd.read_csv(csv_path)

    # Smooth speed with a rolling window for cleaner visualisation
    WINDOW = 10
    df["speed_smooth"] = df["avg_speed"].rolling(WINDOW, min_periods=1, center=True).mean()

    has_fps = "fps" in df.columns
    n_panels = 4 if has_fps else 3
    fig, axes = plt.subplots(n_panels, 1, figsize=(12, 3 * n_panels), sharex=True)
    fig.suptitle("TrafficNet — Per-Frame Analytics", fontsize=14, fontweight="bold")

    # ── Panel 1: active tracks ───────────────────────────────────────────────
    ax1 = axes[0]
    ax1.fill_between(df["frame_id"], df["active_tracks"], alpha=0.4, color="steelblue")
    ax1.plot(df["frame_id"], df["active_tracks"], color="steelblue", linewidth=1.0)
    ax1.set_ylabel("Active tracks")
    ax1.set_title("Vehicle Count Over Time")
    ax1.grid(True, alpha=0.3)

    # ── Panel 2: average speed ───────────────────────────────────────────────
    ax2 = axes[1]
    ax2.plot(df["frame_id"], df["avg_speed"], color="lightgray", linewidth=0.6,
             label="raw")
    ax2.plot(df["frame_id"], df["speed_smooth"], color="darkorange", linewidth=1.5,
             label=f"smoothed (w={WINDOW})")
    ax2.set_ylabel("Avg speed (px / frame)")
    ax2.set_title("Average Vehicle Speed Over Time")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # ── Panel 3: per-frame FPS (optional) ───────────────────────────────────
    if has_fps:
        ax_fps = axes[2]
        fps_smooth = df["fps"].rolling(WINDOW, min_periods=1, center=True).mean()
        ax_fps.plot(df["frame_id"], df["fps"], color="lightgray", linewidth=0.5)
        ax_fps.plot(df["frame_id"], fps_smooth, color="mediumseagreen", linewidth=1.5,
                    label=f"smoothed (w={WINDOW})")
        ax_fps.axhline(15, color="red", linestyle="--", linewidth=1, label="15 FPS target")
        ax_fps.set_ylabel("FPS")
        ax_fps.set_title("Processing Throughput Over Time")
        ax_fps.legend(fontsize=8)
        ax_fps.grid(True, alpha=0.3)

    # ── Panel 4: congestion state ────────────────────────────────────────────
    ax3 = axes[3 if has_fps else 2]
    congested = df["congested"].astype(int)
    ax3.step(df["frame_id"], congested, where="post", color="crimson", linewidth=1.2)
    ax3.fill_between(df["frame_id"], congested, step="post", alpha=0.3, color="crimson")
    ax3.set_yticks([0, 1])
    ax3.set_yticklabels(["Clear", "Congested"])
    ax3.set_ylabel("State")
    ax3.set_xlabel("Frame")
    ax3.set_title("Congestion Detection Over Time")
    ax3.grid(True, alpha=0.3)

    # Shade congested intervals on all panels for context
    congested_frames = df[df["congested"]]["frame_id"].values
    if len(congested_frames) > 0:
        for ax in axes[:(3 if has_fps else 2)]:
            ax.fill_between(df["frame_id"],
                            ax.get_ylim()[0], ax.get_ylim()[1],
                            where=df["congested"].values.astype(bool),
                            alpha=0.08, color="crimson",
                            label="congested interval")

    red_patch = mpatches.Patch(color="crimson", alpha=0.3, label="Congested interval")
    axes[0].legend(handles=[red_patch], fontsize=8, loc="upper right")

    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot → {output_path}")
    if show:
        plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv",           help="Path to metrics CSV from main.py")
    parser.add_argument("--output", "-o", default=None,
                        help="Path to save plot (PNG/PDF). If omitted, displays interactively.")
    parser.add_argument("--no-show", action="store_true",
                        help="Don't open an interactive window (useful for headless runs)")
    args = parser.parse_args()

    if not os.path.isfile(args.csv):
        print(f"Error: file not found: {args.csv}", file=sys.stderr)
        sys.exit(1)

    plot_analytics(args.csv,
                   output_path=args.output,
                   show=not args.no_show)


if __name__ == "__main__":
    main()
