"""
Motion Analyzer Module
======================
Analyzes optical flow magnitude data frame-by-frame to detect abnormal
motion events (sudden motion, abnormal stops) and produces structured
alert payloads that can be forwarded to the backend via Kafka.

This module performs rule-based analysis on the numerical output of the
Optical Flow model. It does NOT run any additional ML inference.
"""

import logging
import os
import numpy as np

logger = logging.getLogger("optical_flow.motion_analyzer")

# ---------------------------------------------------------------------------
# Default thresholds – can be overridden via environment variables
# ---------------------------------------------------------------------------
DEFAULT_SUDDEN_MOTION_MULTIPLIER = float(os.getenv("ALERT_SUDDEN_MOTION_MULTIPLIER", "3.0"))
DEFAULT_STOP_DURATION_THRESHOLD = float(os.getenv("ALERT_STOP_DURATION_SEC", "3.0"))
DEFAULT_STOP_MAGNITUDE_CEILING = float(os.getenv("ALERT_STOP_MAGNITUDE_CEILING", "0.8"))
DEFAULT_HISTORY_WINDOW = int(os.getenv("ALERT_HISTORY_WINDOW", "30"))
DEFAULT_COOLDOWN_SEC = float(os.getenv("ALERT_COOLDOWN_SEC", "5.0"))


class MotionAnalyzer:
    """Frame-by-frame motion analyzer that detects abnormal events.

    Usage::

        analyzer = MotionAnalyzer(fps=30.0)
        for idx, flow in enumerate(frames):
            analyzer.analyze_frame(flow, idx)
        alerts = analyzer.get_alerts()
    """

    def __init__(
        self,
        fps: float = 30.0,
        sudden_motion_multiplier: float = DEFAULT_SUDDEN_MOTION_MULTIPLIER,
        stop_duration_threshold: float = DEFAULT_STOP_DURATION_THRESHOLD,
        stop_magnitude_ceiling: float = DEFAULT_STOP_MAGNITUDE_CEILING,
        history_window: int = DEFAULT_HISTORY_WINDOW,
        cooldown_sec: float = DEFAULT_COOLDOWN_SEC,
        is_moving: bool = True,
    ):
        # Video metadata
        self.fps = max(fps, 1.0)

        # Configurable thresholds
        self.sudden_motion_multiplier = sudden_motion_multiplier
        self.stop_duration_threshold = stop_duration_threshold
        self.stop_magnitude_ceiling = stop_magnitude_ceiling
        self.history_window = max(history_window, 5)
        self.cooldown_sec = cooldown_sec
        self.is_moving = is_moving

        # Internal state
        self._magnitude_history: list[float] = []
        self._alerts: list[dict] = []
        self._low_motion_start_frame: int | None = None
        self._last_alert_frame: int = -9999

        logger.info(
            "MotionAnalyzer initialized fps=%.1f sudden_multiplier=%.1f "
            "stop_duration=%.1fs stop_ceiling=%.2f history_window=%d "
            "cooldown=%.1fs is_moving=%s",
            self.fps,
            self.sudden_motion_multiplier,
            self.stop_duration_threshold,
            self.stop_magnitude_ceiling,
            self.history_window,
            self.cooldown_sec,
            self.is_moving,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_frame(self, flow: np.ndarray, frame_index: int) -> None:
        """Analyze one optical-flow output and check for anomalies.

        Parameters
        ----------
        flow : np.ndarray
            The raw optical flow array (H, W, 2) or compatible shape.
            Channels represent (u, v) displacement vectors.
        frame_index : int
            0-based index of the current frame in the video.
        """
        mean_magnitude = self._compute_mean_magnitude(flow)
        if mean_magnitude is None:
            return

        self._magnitude_history.append(mean_magnitude)

        # --- Check: Sudden Motion (spike relative to recent average) ---
        self._check_sudden_motion(mean_magnitude, frame_index)

        # --- Check: Abnormal Stop (near-zero motion while isMoving) ---
        if self.is_moving:
            self._check_abnormal_stop(mean_magnitude, frame_index)

    def get_alerts(self) -> list[dict]:
        """Return a copy of all detected alerts."""
        return list(self._alerts)

    # ------------------------------------------------------------------
    # Detection logic
    # ------------------------------------------------------------------

    def _check_sudden_motion(self, magnitude: float, frame_index: int) -> None:
        """Detect a spike where the current magnitude is significantly
        above the running average of the last *history_window* frames."""
        if len(self._magnitude_history) < self.history_window:
            return  # Not enough history yet

        if self._in_cooldown(frame_index):
            return

        window = self._magnitude_history[-self.history_window - 1 : -1]
        if not window:
            return

        avg = float(np.mean(window))
        threshold = avg * self.sudden_motion_multiplier

        # Only trigger if threshold is meaningful (avoid false positives
        # when the scene is nearly static and avg ≈ 0).
        if avg < 0.3:
            return

        if magnitude > threshold:
            timestamp_sec = round(frame_index / self.fps, 2)
            alert = {
                "type": "sudden_motion",
                "message": (
                    f"Phát hiện chuyển động đột ngột tại giây thứ {timestamp_sec} "
                    f"của video (magnitude={magnitude:.1f}, avg={avg:.1f})"
                ),
                "severity": "HIGH",
                "timestamp_sec": timestamp_sec,
                "frame_index": frame_index,
                "magnitude": round(float(magnitude), 2),
            }
            self._alerts.append(alert)
            self._last_alert_frame = frame_index
            logger.info(
                "ALERT sudden_motion frame=%d ts=%.2fs mag=%.1f avg=%.1f threshold=%.1f",
                frame_index,
                timestamp_sec,
                magnitude,
                avg,
                threshold,
            )

    def _check_abnormal_stop(self, magnitude: float, frame_index: int) -> None:
        """Detect an abnormal stop: magnitude stays near-zero for longer
        than *stop_duration_threshold* seconds while device reports moving."""
        if magnitude < self.stop_magnitude_ceiling:
            if self._low_motion_start_frame is None:
                self._low_motion_start_frame = frame_index
            else:
                duration_frames = frame_index - self._low_motion_start_frame
                duration_sec = duration_frames / self.fps
                if duration_sec >= self.stop_duration_threshold:
                    if not self._in_cooldown(frame_index):
                        timestamp_sec = round(self._low_motion_start_frame / self.fps, 2)
                        alert = {
                            "type": "abnormal_stop",
                            "message": (
                                f"Phát hiện dừng xe bất thường kéo dài {duration_sec:.1f}s "
                                f"bắt đầu từ giây thứ {timestamp_sec} của video"
                            ),
                            "severity": "MEDIUM",
                            "timestamp_sec": timestamp_sec,
                            "frame_index": self._low_motion_start_frame,
                            "magnitude": round(float(magnitude), 2),
                        }
                        self._alerts.append(alert)
                        self._last_alert_frame = frame_index
                        logger.info(
                            "ALERT abnormal_stop frame=%d start=%d duration=%.1fs mag=%.2f",
                            frame_index,
                            self._low_motion_start_frame,
                            duration_sec,
                            magnitude,
                        )
                    # Reset so we don't keep firing every frame
                    self._low_motion_start_frame = frame_index
        else:
            # Motion resumed → reset tracker
            self._low_motion_start_frame = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_mean_magnitude(self, flow: np.ndarray) -> float | None:
        """Compute mean magnitude from an optical-flow tensor."""
        try:
            arr = np.asarray(flow, dtype=np.float32)
        except Exception:
            return None

        # Handle various shapes that inference.py may produce
        if arr.ndim == 3 and arr.shape[2] == 2:
            # (H, W, 2) – most common after postprocess_flow
            u, v = arr[:, :, 0], arr[:, :, 1]
        elif arr.ndim == 3 and arr.shape[0] == 2:
            # (2, H, W)
            u, v = arr[0], arr[1]
        elif arr.ndim == 4 and arr.shape[1] == 2:
            # (1, 2, H, W)
            u, v = arr[0, 0], arr[0, 1]
        elif arr.ndim == 4 and arr.shape[3] == 2:
            # (1, H, W, 2)
            u, v = arr[0, :, :, 0], arr[0, :, :, 1]
        else:
            logger.warning("Unsupported flow shape for magnitude calculation: %s", arr.shape)
            return None

        u = np.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)

        magnitude = np.sqrt(u ** 2 + v ** 2)
        return float(np.mean(magnitude))

    def _in_cooldown(self, frame_index: int) -> bool:
        """Return True if we are still in cooldown from the last alert."""
        cooldown_frames = int(self.cooldown_sec * self.fps)
        return (frame_index - self._last_alert_frame) < cooldown_frames
