import unittest
import numpy as np
from motion_analyzer import MotionAnalyzer

class TestMotionAnalyzer(unittest.TestCase):
    def setUp(self):
        # Create a standard analyzer with 10 FPS to make duration math simple
        self.fps = 10.0
        self.analyzer = MotionAnalyzer(
            fps=self.fps,
            sudden_motion_multiplier=3.0,
            stop_duration_threshold=2.0,  # 2 seconds = 20 frames
            stop_magnitude_ceiling=0.5,
            history_window=10,
            cooldown_sec=1.0,
            is_moving=True
        )

    def generate_flow_frame(self, magnitude_value, shape=(20, 20, 2)):
        """Helper to generate a mock optical flow frame with a uniform magnitude."""
        # u^2 + v^2 = magnitude_value^2
        # If u = v = magnitude_value / sqrt(2)
        val = magnitude_value / np.sqrt(2)
        return np.full(shape, val, dtype=np.float32)

    def test_initialization(self):
        self.assertEqual(self.analyzer.fps, self.fps)
        self.assertEqual(self.analyzer.sudden_motion_multiplier, 3.0)
        self.assertEqual(self.analyzer.stop_duration_threshold, 2.0)
        self.assertEqual(self.analyzer.stop_magnitude_ceiling, 0.5)
        self.assertEqual(self.analyzer.is_moving, True)
        self.assertEqual(len(self.analyzer.get_alerts()), 0)

    def test_normal_movement_no_alerts(self):
        # Feed 30 frames of steady movement (magnitude = 2.0)
        flow_frame = self.generate_flow_frame(2.0)
        for i in range(30):
            self.analyzer.analyze_frame(flow_frame, i)
        
        self.assertEqual(len(self.analyzer.get_alerts()), 0)

    def test_sudden_motion_alert(self):
        # 1. Feed 10 frames of low baseline motion (magnitude = 1.0)
        low_flow = self.generate_flow_frame(1.0)
        for i in range(10):
            self.analyzer.analyze_frame(low_flow, i)
            
        # 2. Feed a sudden motion frame (magnitude = 4.0, which is > 3.0 * baseline)
        spike_flow = self.generate_flow_frame(4.5)
        self.analyzer.analyze_frame(spike_flow, 10)
        
        alerts = self.analyzer.get_alerts()
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0]["type"], "sudden_motion")
        self.assertEqual(alerts[0]["frame_index"], 10)
        self.assertEqual(alerts[0]["timestamp_sec"], 1.0)  # 10 / 10 FPS = 1.0s
        self.assertEqual(alerts[0]["severity"], "HIGH")

    def test_abnormal_stop_alert(self):
        # 1. Feed 5 frames of normal baseline movement (magnitude = 2.0)
        normal_flow = self.generate_flow_frame(2.0)
        for i in range(5):
            self.analyzer.analyze_frame(normal_flow, i)
            
        # 2. Suddenly stop: feed 21 frames of near-zero motion (magnitude = 0.2)
        # stop_duration_threshold is 2.0s = 20 frames. 21 frames should trigger.
        stopped_flow = self.generate_flow_frame(0.2)
        for i in range(5, 27):
            self.analyzer.analyze_frame(stopped_flow, i)
            
        alerts = self.analyzer.get_alerts()
        # Should have detected the abnormal stop
        self.assertTrue(any(a["type"] == "abnormal_stop" for a in alerts))
        
        # Find the abnormal stop alert
        stop_alert = next(a for a in alerts if a["type"] == "abnormal_stop")
        # The stop starts at frame 5
        self.assertEqual(stop_alert["frame_index"], 5)
        self.assertEqual(stop_alert["timestamp_sec"], 0.5)  # 5 / 10 FPS = 0.5s

    def test_cooldown_mechanism(self):
        # Feed baseline
        normal_flow = self.generate_flow_frame(1.0)
        for i in range(10):
            self.analyzer.analyze_frame(normal_flow, i)
            
        # Trigger first sudden motion
        spike_flow = self.generate_flow_frame(4.5)
        self.analyzer.analyze_frame(spike_flow, 10)
        
        # Try to trigger another sudden motion immediately at frame 12 (within 1.0s / 10 frames cooldown)
        self.analyzer.analyze_frame(spike_flow, 12)
        
        alerts = self.analyzer.get_alerts()
        # Cooldown should prevent the second alert from firing
        self.assertEqual(len(alerts), 1)

if __name__ == "__main__":
    unittest.main()
