import argparse
import time

from detection.detector import Detector
from tracking.tracker import Tracker
from analytics.counting import count_crossings, flow_rate
from analytics.lanes import assign_lanes, lane_utilization
from analytics.congestion import CongestionDetector
from visualization.draw import (
    draw_tracks, draw_trails, draw_counting_lines,
    draw_lane_rois, draw_congestion,
)
from utils.io import load_frames, save_video, save_metrics_csv
from config import COUNTING_LINES, LANE_ROIS, CONF_THRESHOLD


def run(source, output_video=None, output_metrics=None, show=False, delay=0):
    detector   = Detector()
    tracker    = Tracker()

    per_frame_metrics = []
    annotated_frames  = []
    congestion_detector = None  # initialized on first frame
    frame_times = []

    for frame_id, frame in load_frames(source):
        _t0 = time.perf_counter()

        # Perception 
        detections = detector.detect_frame_bgr(frame)

        # Tracking 
        track_outputs = tracker.update(frame_id, detections)

        # Congestion (needs frame dimensions on first call)
        if congestion_detector is None:
            h, w = frame.shape[:2]
            congestion_detector = CongestionDetector(w, h)
        congestion_detector.update(tracker.tracks)

        # Per-frame metrics
        status = congestion_detector.status()
        per_frame_metrics.append({
            "frame_id":      frame_id,
            "active_tracks": len(tracker.tracks),
            "avg_speed":     status["avg_speed"],
            "density":       status["density"],
            "congested":     status["congested"],
            "fps":           round(1.0 / frame_times[-1], 2) if frame_times else 0.0,
        })

        # Visualization 
        crossings = count_crossings(tracker.trajectory_store, COUNTING_LINES)
        util = lane_utilization(assign_lanes(tracker.trajectory_store, LANE_ROIS))

        draw_lane_rois(frame, LANE_ROIS, util)
        draw_counting_lines(frame, COUNTING_LINES, crossings)
        draw_trails(frame, tracker.trajectory_store)
        draw_tracks(frame, track_outputs)
        draw_congestion(frame, status)

        annotated_frames.append(frame)
        frame_times.append(time.perf_counter() - _t0)

        if show:
            import cv2
            cv2.imshow("TrafficNet", frame)
            key = cv2.waitKey(delay) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):   # spacebar pauses until next keypress
                cv2.waitKey(0)

        fps_inst = 1.0 / frame_times[-1] if frame_times[-1] > 0 else 0.0
        print(f"Frame {frame_id:04d} | tracks={len(tracker.tracks):2d} | "
              f"speed={status['avg_speed']:5.1f} px/fr | "
              f"{'CONGESTED' if status['congested'] else 'clear':9s} | "
              f"{fps_inst:5.1f} FPS")

    if show:
        import cv2
        cv2.destroyAllWindows()

    # Summary analytics
    store     = tracker.trajectory_store
    crossings = count_crossings(store, COUNTING_LINES)
    rate      = flow_rate(crossings, num_frames=len(annotated_frames))
    util      = lane_utilization(assign_lanes(store, LANE_ROIS))

    # Throughput stats
    if frame_times:
        avg_fps  = len(frame_times) / sum(frame_times)
        med_fps  = 1.0 / sorted(frame_times)[len(frame_times) // 2]
        p5_fps   = 1.0 / sorted(frame_times)[int(len(frame_times) * 0.95)]  # 5th-percentile fps
        total_s  = sum(frame_times)

    summary = {
        "total_crossings":   {n: d["count"] for n, d in crossings.items()},
        "flow_rate_veh_min": rate,
        "lane_utilization":  util,
        "avg_fps":           round(avg_fps, 1) if frame_times else 0.0,
    }

    print("\n" + "=" * 52)
    print("  TRAFFIC ANALYTICS SUMMARY")
    print("=" * 52)
    if frame_times:
        print(f"  Frames processed : {len(frame_times)}")
        print(f"  Total wall time  : {total_s:.1f} s")
        print(f"  Avg throughput   : {avg_fps:.1f} FPS")
        print(f"  Median FPS       : {med_fps:.1f} FPS")
        print(f"  Worst-5%% FPS    : {p5_fps:.1f} FPS")
        print(f"  Proposal target  : ≥ 15 FPS  "
              f"{'✓' if avg_fps >= 15 else '✗ (below target)'}")
        print()
    print(f"  {'Counting line':<16} {'Crossings':>10}  {'Flow (veh/min)':>14}")
    print(f"  {'-'*16}  {'-'*10}  {'-'*14}")
    for line_name, count in summary["total_crossings"].items():
        rate_val = summary["flow_rate_veh_min"][line_name]
        print(f"  {line_name:<16} {count:>10}  {rate_val:>14.2f}")

    total_vehicles = sum(summary["total_crossings"].values())
    print(f"\n  Total vehicle crossings: {total_vehicles}")

    print(f"\n  {'Lane / Arm':<18} {'Utilization':>12}")
    print(f"  {'-'*18}  {'-'*12}")
    for lane, frac in sorted(util.items(), key=lambda kv: -kv[1]):
        pct = frac * 100
        bar = "█" * int(pct / 5)
        print(f"  {lane:<18} {pct:>10.1f}%  {bar}")
    print("=" * 52)

    # Save outputs
    if output_video:
        save_video(annotated_frames, output_video)

    if output_metrics:
        save_metrics_csv(per_frame_metrics, summary, output_metrics)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TrafficNet — vision-based traffic analytics")
    parser.add_argument("source",         help="Image folder or video file path")
    parser.add_argument("--output-video",   default=None, help="Path to save annotated video")
    parser.add_argument("--output-metrics", default=None, help="Path to save metrics CSV")
    parser.add_argument("--show",  action="store_true", help="Display frames in a window")
    parser.add_argument("--delay", type=int, default=30,
                        help="ms to wait between frames when --show is used (0 = wait for keypress)")
    args = parser.parse_args()

    run(args.source,
        output_video=args.output_video,
        output_metrics=args.output_metrics,
        show=args.show,
        delay=args.delay)
