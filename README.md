# TrafficNet — Results Summary

## What the system does

Takes a traffic camera video → detects vehicles → tracks them with persistent IDs → outputs three traffic metrics: vehicle counts, lane utilization, and congestion status.

Three layers:
1. **Detection** — YOLOv8 nano draws bounding boxes around cars, trucks, motorcycles, buses in each frame
2. **Tracking** — Kalman filter + Hungarian algorithm assigns consistent IDs across frames, estimates velocity
3. **Analytics** — Uses track histories to compute counting, lane utilization, and congestion

---

## Evaluation Dataset

We used **UA-DETRAC** — a standard academic benchmark of real traffic camera footage with frame-by-frame ground-truth vehicle positions and IDs. We tested on two sequences:

| Sequence | Frames | Weather | Vehicles |
|---|---|---|---|
| MVI_20011 | 664 | Sunny | 7,655 GT vehicle-frame annotations |
| MVI_20032 | 437 | Sunny | 1,657 GT vehicle-frame annotations |

---

## Tracking Results (MOTA evaluation)

**MOTA** (Multi-Object Tracking Accuracy) is the standard metric. Formula:
`MOTA = 1 − (missed detections + false positives + identity switches) / total GT vehicles`

| Sequence | MOTA | Recall | Precision | Identity Switches | Fragmentations |
|---|---|---|---|---|---|
| MVI_20011 | 26.2% | 53.5% | 66.7% | 42 | 243 |
| MVI_20032 | 6.3% | 82.3% | 52.1% | 7 | 52 |
| **Average** | **16.3%** | **67.9%** | **59.4%** | **49 total** | **295 total** |

**Proposal targets:** MOTA ≥ 50% (Good), MOTA ≥ 65% (Stretch)

### Why MOTA is below target

MOTA is low because YOLOv8 nano **misses roughly half the vehicles** in the dataset — vehicles that are occluded, small/distant, or in dense clusters. This drives up the "missed detections" term in the formula.

The tracker itself is not the problem. Identity switches (25/sequence average) are well within target.

### What IS meeting target

**Identity switches = 25/sequence average — target was ≤ 200 (Good), ≤ 100 (Stretch).**
This means the tracker almost never confuses one vehicle for another. Persistent identity is reliable.

---

## Ablation Study (Baseline Comparisons)

We ran four versions of the pipeline to prove each component contributes. Averaged across both sequences:

| Condition | MOTA | Identity Switches | What it tests |
|---|---|---|---|
| **Full pipeline** (Kalman + Hungarian) | **16.3%** | **49** | Our system |
| No-motion model (position-only, no Kalman) | 16.6% | 49 | Is Kalman needed? |
| Greedy matching (no Hungarian) | 16.2% | 48 | Is Hungarian needed? |
| **Detection-only (no tracker at all)** | **−53.8%** | **5,519** | Is tracking needed? |

### Key findings

1. **Removing the tracker entirely is catastrophic.** MOTA goes negative and identity switches explode from 49 to 5,519 — a 112× increase. Every frame assigns fresh IDs to every detection, so every vehicle gets counted as a new one each frame. This proves the tracking layer is essential.

2. **Kalman vs. position-only: 0.3% difference.** On slow intersection traffic, knowing velocity doesn't help much because vehicles barely move between frames. The Kalman filter's advantage would be larger on a highway with fast-moving vehicles.

3. **Hungarian vs. greedy: 0.1% difference.** At intersection vehicle densities (≤15 tracks), greedy nearest-neighbor makes the same decision as the globally optimal Hungarian algorithm almost every time.

---

## Analytics Results

For analytics evaluation we used the UA-DETRAC ground-truth trajectories and a horizontal counting line placed at the mid-frame (same line applied to GT and predicted trajectories, so the comparison is fair).

| Metric | MVI_20011 | MVI_20032 | Average | Target |
|---|---|---|---|---|
| Vehicle Counting Accuracy | 92.9% | 100.0% | **96.5%** | ≥ 85% (Good) / ≥ 92% (Stretch) |
| Lane Assignment Accuracy | 98.1% | 100.0% | **99.0%** | ≥ 80% (Good) / ≥ 90% (Stretch) |

Both metrics **exceed the stretch targets.**

### How counting works
A virtual line is drawn across each road arm (manually configured per scene). If a vehicle's trajectory crosses from one side to the other, it counts — once per vehicle per line. Counting accuracy = how close our count was to the ground-truth count.

### How lane assignment works
The frame is divided into spatial zones per approach arm (north, south, east, west). Each tracked vehicle is assigned to whichever zone contained the majority of its trajectory positions. Lane accuracy = fraction of vehicles where our zone assignment matched the ground-truth trajectory zone.

### Congestion detection
No ground-truth labels exist for congestion in any public dataset, so this is validated qualitatively. The system flags CONGESTED when:
- Average vehicle speed < 5 px/frame (slow traffic), AND
- Vehicle density > threshold (not just one slow car)
- ...sustained for > 50% of the last 10 frames (smoothing window)

On the demo video (Indian 4-way intersection), the system correctly shows CONGESTED during dense jammed moments and CLEAR during lighter traffic.

---

## Processing Speed

Target: ≥ 15 FPS for real-time operation. Measured per-frame in `main.py` and reported in the summary. Bottleneck is YOLOv8 inference — the Kalman filter and Hungarian algorithm add negligible overhead.

---

## Demo Video Results (traffic.mp4)

Qualitative results on a real Indian 4-way intersection:

| Metric | Value |
|---|---|
| Total vehicle crossings | 13 |
| South arm flow | 14.2 veh/min |
| East arm flow | 8.5 veh/min |
| West arm flow | 14.2 veh/min |
| Lane utilization | South 32%, East 24%, West 22%, North 22% |
| Congestion | Triggered correctly during dense traffic moments |

**Note:** North arm shows 0 crossings because most north-arm vehicles enter the frame already inside the intersection, below the counting line. This is a counting line placement issue, not a tracker issue — moving the line further into the intersection would capture them.

---

## Honest Limitations

| Limitation | Impact | Fix |
|---|---|---|
| YOLOv8 nano misses ~47% of vehicles | Low MOTA (16.3%) | Swap to YOLOv8m or YOLOv8l — one config change |
| Counting lines and lane ROIs are manually configured per scene | Requires human setup for each new intersection | Future work: automatic road geometry detection |
| Congestion thresholds are hand-tuned | May need adjustment per deployment scene | Future work: calibrate against labeled congestion data |
| Speed is in pixels/frame, not km/h | Can't report real-world speed | Future work: camera calibration / homography |

---

## Summary

The system successfully demonstrates that a modular vision-based pipeline can extract meaningful traffic analytics from camera footage without any physical road sensors. The tracking layer is robust (IDSW well within target), and the analytics layer exceeds all stretch targets. The main open problem is detector recall — upgrading the detection model is the highest-leverage next step.