"""
engine/live_qr/tracker.py
=========================
QR code multi-object tracker with instantaneous velocity prediction.

Algorithm
---------
Each active QR code gets a stable integer ID permanently bound to its own
velocity vector.  The velocity is derived from a sliding window of the
**two most recent consecutive detections** and scaled by VELOCITY_FACTOR
(0.15) to produce a conservative prediction offset.

Velocity formula (per Cartesian axis, sign preserved):

    vx = ((x_final - x_initial) / Δt) × (±0.15)
    vy = ((y_final - y_initial) / Δt) × (±0.15)

Predicted position at time t:

    x_pred(t) = x_last + vx × (t - t_last)
    y_pred(t) = y_last + vy × (t - t_last)

Per-frame pipeline:
  1. Predict  — advance each track to current timestamp using its velocity.
  2. Match    — greedy nearest-predicted-position assignment (dynamic gate).
  3. Update   — append detection to history; recalculate velocity if consecutive.
  4. Birth    — unmatched detections → new tracks.
  5. Expire   — tracks absent > max_missed_s → deleted.
  6. Ghost    — surviving-but-unmatched tracks → GhostDetection list for UI overlay.

Key properties
--------------
* Velocity recalculated only from **consecutive** detections (missed == 0
  before the new hit).  During a gap the last velocity drives prediction.
* Gate radius expands per missed frame: base + 35 × missed (px).
* Both matched (TrackedDetection) and coasting (GhostDetection) tracks
  carry their predicted centroid so the UI can visualise the trajectory.

Dependencies: NumPy only.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from app.src.engine.modules.decoding.live_qr.detector import Detection


# ── Published types ────────────────────────────────────────────────────────────

@dataclass
class TrackedDetection:
    """YOLO Detection enriched with a stable tracking ID and velocity prediction."""

    detection:  Detection
    track_id:   int
    pred_cx:    float   # 10-frame-ahead ghost centroid x (visualization)
    pred_cy:    float   # 10-frame-ahead ghost centroid y (visualization)
    vel_mag:    float = 0.0   # velocity magnitude in px/s — drives duo-read decision
    pred1_cx:   float = 0.0   # 1-frame-ahead centroid x — for decoder pre-fetch
    pred1_cy:   float = 0.0   # 1-frame-ahead centroid y

    @property
    def x1(self) -> int:           return self.detection.x1
    @property
    def y1(self) -> int:           return self.detection.y1
    @property
    def x2(self) -> int:           return self.detection.x2
    @property
    def y2(self) -> int:           return self.detection.y2
    @property
    def width(self) -> int:        return self.detection.width
    @property
    def height(self) -> int:       return self.detection.height
    @property
    def confidence(self) -> float: return self.detection.confidence


@dataclass
class GhostDetection:
    """
    Velocity prediction for a coasting track that had no detection this frame.

    Carries enough geometry to render a predicted bbox in the UI overlay.
    No real Detection exists — only the tracker's estimate of where the QR is.
    """
    track_id:   int
    pred_cx:    float
    pred_cy:    float
    est_width:  float   # last known bbox width  (px)
    est_height: float   # last known bbox height (px)


# ── Tuning constants ───────────────────────────────────────────────────────────

# Fraction of raw instantaneous velocity applied to prediction.
# 0.15 → predicts 15 % of the measured Δx/Δt per frame interval.
_VELOCITY_FACTOR  = 0.15

# Number of past centroids kept per track.
_HISTORY_N        = 5

# Matching gate: base + expansion per missed frame.
_BASE_GATE_PX     = 200.0
_MISSED_EXPAND_PX = 35.0

# Adaptive EMA for velocity blending.
# SMOOTH: low alpha → prediction stable for consistent movement / jitter rejection.
# SNAP  : high alpha → fast reaction when direction reverses (dot product < 0).
_EMA_ALPHA_SMOOTH  = 0.30
_EMA_ALPHA_SNAP    = 0.85

# Maximum scaled velocity (px/s after VELOCITY_FACTOR). Clamps spurious spikes.
# Equivalent to ≈ 600 px/s of raw displacement — covers fast hand motion.
_MAX_VELOCITY_PX_S = 90.0


# ── Single-track velocity model ────────────────────────────────────────────────

class _VelocityTrack:
    """
    One QR code: position history, velocity bound to the ID, and bbox dimensions.

    Velocity recalculation uses a sliding window of the 2 most recent
    *consecutive* detections.  If the previous frame was missed, the old
    velocity is preserved for continued prediction without update.
    """

    def __init__(self, tid: int, cx: float, cy: float,
                 w: float, h: float, ts: float) -> None:
        self.id          = tid
        self._hist: List[Tuple[float, float, float]] = [(cx, cy, ts)]
        self._vx: float  = 0.0   # px/s, scaled by VELOCITY_FACTOR
        self._vy: float  = 0.0
        self.last_w      = w     # last observed bbox width
        self.last_h      = h     # last observed bbox height
        self.missed      = 0
        self.hits        = 1
        self._ts_matched = ts

    # ── Velocity ───────────────────────────────────────────────────────────────

    def _recalc_velocity(self) -> None:
        """
        Adaptive EMA velocity update.

        1. Compute raw instantaneous velocity from the 2 most recent history
           entries and scale it by VELOCITY_FACTOR.
        2. Choose blending alpha based on the dot product with the previous
           velocity vector:
             dot > 0 → same general direction → smooth (alpha = 0.30)
             dot < 0 → direction reversed     → snap   (alpha = 0.85)
        3. Clamp the result to _MAX_VELOCITY_PX_S to reject spurious spikes.
        """
        x2, y2, t2 = self._hist[-1]
        x1, y1, t1 = self._hist[-2]
        dt = max(t2 - t1, 1.0 / 120.0)

        raw_vx = ((x2 - x1) / dt) * _VELOCITY_FACTOR
        raw_vy = ((y2 - y1) / dt) * _VELOCITY_FACTOR

        dot   = self._vx * raw_vx + self._vy * raw_vy
        alpha = _EMA_ALPHA_SNAP if dot < 0 else _EMA_ALPHA_SMOOTH

        bvx = alpha * raw_vx + (1.0 - alpha) * self._vx
        bvy = alpha * raw_vy + (1.0 - alpha) * self._vy

        mag = (bvx * bvx + bvy * bvy) ** 0.5
        if mag > _MAX_VELOCITY_PX_S:
            s    = _MAX_VELOCITY_PX_S / mag
            bvx *= s
            bvy *= s

        self._vx = bvx
        self._vy = bvy

    # ── Core methods ──────────────────────────────────────────────────────────

    def update(self, cx: float, cy: float, w: float, h: float, ts: float) -> None:
        """Register a matched detection; recalculate velocity if consecutive."""
        was_consecutive = (self.missed == 0)

        self._hist.append((cx, cy, ts))
        if len(self._hist) > _HISTORY_N:
            self._hist.pop(0)

        if was_consecutive and len(self._hist) >= 2:
            self._recalc_velocity()

        self.last_w      = w
        self.last_h      = h
        self.missed      = 0
        self.hits       += 1
        self._ts_matched = ts

    def _dt_frame(self) -> float:
        """Estimated inter-frame interval from the two most recent history entries."""
        if len(self._hist) >= 2:
            return max(self._hist[-1][2] - self._hist[-2][2], 1.0 / 120.0)
        return 1.0 / 20.0   # cold start: assume 20 fps

    def predict(self, ts: float) -> Tuple[float, float]:
        """Position at absolute time `ts` (used for matching gate)."""
        x_last, y_last, t_last = self._hist[-1]
        dt = max(ts - t_last, 0.0)
        return x_last + self._vx * dt, y_last + self._vy * dt

    def predict_next_1frame(self) -> Tuple[float, float]:
        """Predicted centroid exactly 1 estimated frame ahead (for decoder pre-fetch)."""
        dt = self._dt_frame()
        x_last, y_last, _ = self._hist[-1]
        return x_last + self._vx * dt, y_last + self._vy * dt

    def predict_next(self) -> Tuple[float, float]:
        """
        Predicted position TWO estimated frames ahead of the most recent detection.

        One frame ahead = where YOLO will draw next.
        Two frames ahead = one step beyond that — the ghost leads the detector.

            x_ghost = x_last + vx × (2 × dt_frame)
        """
        dt = self._dt_frame() * 10
        x_last, y_last, _ = self._hist[-1]
        return x_last + self._vx * dt, y_last + self._vy * dt

    def predict_next_from(self, now: float) -> Tuple[float, float]:
        """
        Two estimated frames ahead of `now` (coasting ghost bboxes).

        Keeps the ghost leading the trajectory even when no detection arrives.
        """
        dt = self._dt_frame() * 10
        return self.predict(now + dt)

    def gate_radius(self) -> float:
        return _BASE_GATE_PX + _MISSED_EXPAND_PX * self.missed

    @property
    def last_pos(self) -> Tuple[float, float]:
        x, y, _ = self._hist[-1]
        return x, y

    @property
    def velocity(self) -> Tuple[float, float]:
        return self._vx, self._vy


# ── Multi-object tracker ───────────────────────────────────────────────────────

# Return type: matched detections + ghost predictions for coasting tracks.
TrackerResult = Tuple[List[TrackedDetection], List[GhostDetection]]


class QrTracker:
    """
    Manages a pool of _VelocityTrack instances, one per active QR code.

    Returns
    -------
    (detections, ghosts) where:
      detections — one TrackedDetection per YOLO bbox, carrying pred_cx/pred_cy
      ghosts     — one GhostDetection per coasting track (no bbox this frame)

    Both lists are consumed by the UI to render real boxes and predicted overlays.

    Parameters
    ----------
    max_missed_s : float
        Seconds a track survives without a match (velocity predicts through gap).
    """

    def __init__(self, max_missed_s: float = 0.6) -> None:
        self._max_missed_s = max_missed_s
        self._next_id      = 1
        self._tracks: Dict[int, _VelocityTrack] = {}

    # ── Public API ─────────────────────────────────────────────────────────────

    def update(
        self,
        detections: List[Detection],
        ts: Optional[float] = None,
    ) -> TrackerResult:
        """
        Process one frame of YOLO detections.

        Pipeline: record predictions → match → update/birth → expire → ghosts.
        """
        now = ts if ts is not None else time.monotonic()

        track_ids = list(self._tracks.keys())

        # 1. Snapshot predicted positions for ALL tracks before any update ──────
        # These are used both for matching (gate) and for attaching to results.
        predictions: Dict[int, Tuple[float, float]] = {
            tid: self._tracks[tid].predict(now)
            for tid in track_ids
        }
        # Last known positions (no velocity offset) — fallback for abrupt reversals.
        last_positions: Dict[int, Tuple[float, float]] = {
            tid: self._tracks[tid].last_pos
            for tid in track_ids
        }

        # 2. Build Euclidean distance matrix ────────────────────────────────────
        # Dual-hypothesis: min(d_predicted, d_last_known).
        # When a QR reverses suddenly, velocity prediction points the wrong way
        # (d_pred spikes) but d_last_known stays small — the min catches the match.
        nd, nt    = len(detections), len(track_ids)
        assignments: Dict[int, int] = {}

        if nd and nt:
            dist_m = np.full((nd, nt), np.inf)

            for i, det in enumerate(detections):
                cx = (det.x1 + det.x2) / 2.0
                cy = (det.y1 + det.y2) / 2.0
                for j, tid in enumerate(track_ids):
                    px, py = predictions[tid]
                    lx, ly = last_positions[tid]
                    d_pred = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
                    d_last = ((cx - lx) ** 2 + (cy - ly) ** 2) ** 0.5
                    dist_m[i, j] = min(d_pred, d_last)

            # Greedy: sort by distance, assign closest pairs within dynamic gate
            candidates = [
                (dist_m[i, j], i, j)
                for i in range(nd)
                for j, tid in enumerate(track_ids)
                if dist_m[i, j] < self._tracks[tid].gate_radius()
            ]
            candidates.sort()

            matched_d: set[int] = set()
            matched_t: set[int] = set()
            for _, i, j in candidates:
                tid = track_ids[j]
                if i not in matched_d and tid not in matched_t:
                    assignments[i] = tid
                    matched_d.add(i)
                    matched_t.add(tid)

        # 3. Update matched tracks; birth unmatched detections ──────────────────
        result:       List[TrackedDetection] = []
        assigned_ids: set[int]               = set(assignments.values())

        for i, det in enumerate(detections):
            cx = (det.x1 + det.x2) / 2.0
            cy = (det.y1 + det.y2) / 2.0
            w  = float(det.width)
            h  = float(det.height)

            if i in assignments:
                tid = assignments[i]
                # Update FIRST so velocity is refreshed from this detection,
                # then predict_next() gives the next frame — ahead of YOLO box.
                self._tracks[tid].update(cx, cy, w, h, now)
                pred_cx, pred_cy   = self._tracks[tid].predict_next()
                p1cx, p1cy         = self._tracks[tid].predict_next_1frame()
                vx, vy             = self._tracks[tid].velocity
                vel_mag            = (vx * vx + vy * vy) ** 0.5
            else:
                tid = self._next_id
                self._next_id += 1
                self._tracks[tid] = _VelocityTrack(tid, cx, cy, w, h, now)
                assigned_ids.add(tid)
                pred_cx, pred_cy = cx, cy   # new track — no velocity yet
                p1cx, p1cy       = cx, cy
                vel_mag          = 0.0

            result.append(TrackedDetection(det, tid, pred_cx, pred_cy, vel_mag, p1cx, p1cy))

        # 4. Increment missed; expire stale tracks; collect ghosts ──────────────
        ghosts: List[GhostDetection] = []

        for tid in list(self._tracks.keys()):
            if tid not in assigned_ids:
                t = self._tracks[tid]
                t.missed += 1
                if now - t._ts_matched > self._max_missed_s:
                    del self._tracks[tid]
                else:
                    # Project one frame AHEAD of now so the ghost leads, not lags.
                    px, py = t.predict_next_from(now)
                    ghosts.append(GhostDetection(
                        track_id   = tid,
                        pred_cx    = px,
                        pred_cy    = py,
                        est_width  = t.last_w,
                        est_height = t.last_h,
                    ))

        return result, ghosts

    def reset(self) -> None:
        """Remove all tracks and reset ID counter to 1."""
        self._tracks.clear()
        self._next_id = 1
