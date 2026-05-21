"""
ByteTrack implementation following the official ByteTrack paper.
Two-stage association: high-confidence detections matched first,
then low-confidence detections matched with remaining tracks.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import deque


class KalmanFilter:
    """Kalman Filter for tracking bounding boxes with constant velocity model."""

    def __init__(self):
        # State: [x, y, w, h, vx, vy, vw, vh]
        self.F = np.eye(8)
        self.F[0:4, 4:8] = np.eye(4)
        self.H = np.eye(4, 8)
        self.Q = np.eye(8) * 0.01
        self.R = np.eye(4) * 0.1
        self.P = np.eye(8) * 10.0

    def init_state(self, bbox):
        x, y, w, h = bbox
        return np.array([x, y, w, h, 0, 0, 0, 0], dtype=np.float64)

    def predict(self, state, P):
        state = self.F @ state
        P = self.F @ P @ self.F.T + self.Q
        return state, P

    def update(self, state, P, measurement):
        y = measurement - self.H @ state
        S = self.H @ P @ self.H.T + self.R
        K = P @ self.H.T @ np.linalg.inv(S)
        state = state + K @ y
        P = (np.eye(8) - K @ self.H) @ P
        return state, P


class Track:
    """Represents a single tracked object."""

    def __init__(self, track_id, bbox, confidence):
        self.track_id = track_id
        self.kf = KalmanFilter()
        self.state = self.kf.init_state(bbox)
        self.covariance = np.eye(8) * 10.0
        self.confidence = confidence
        self.hits = 1
        self.time_since_update = 0
        self.history = deque(maxlen=30)
        self.history.append(bbox)

    def predict(self):
        self.state, self.covariance = self.kf.predict(self.state, self.covariance)
        self.time_since_update += 1

    def update(self, bbox, confidence):
        self.state, self.covariance = self.kf.update(
            self.state, self.covariance, np.array(bbox)
        )
        self.confidence = confidence
        self.hits += 1
        self.time_since_update = 0
        self.history.append(bbox)

    def get_bbox(self):
        return self.state[:4]


class ByteTrack:
    """
    ByteTrack: Multi-Object Tracking by Associating Every Detection.
    
    Two-stage association:
    Stage 1: Match high-score detections with all tracks
    Stage 2: Match low-score detections with remaining tracks
    """

    def __init__(
        self,
        track_thresh=0.5,
        match_thresh=0.8,
        low_thresh=0.1,
        max_age=30,
        min_hits=3,
    ):
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.low_thresh = low_thresh
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks = []
        self.next_id = 1

    @staticmethod
    def iou(bbox1, bbox2):
        """IoU between two boxes [x, y, w, h] (center format)."""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        x1_min, y1_min = x1 - w1 / 2, y1 - h1 / 2
        x1_max, y1_max = x1 + w1 / 2, y1 + h1 / 2
        x2_min, y2_min = x2 - w2 / 2, y2 - h2 / 2
        x2_max, y2_max = x2 + w2 / 2, y2 + h2 / 2

        inter_x1 = max(x1_min, x2_min)
        inter_y1 = max(y1_min, y2_min)
        inter_x2 = min(x1_max, x2_max)
        inter_y2 = min(y1_max, y2_max)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        area1 = w1 * h1
        area2 = w2 * h2
        union_area = area1 + area2 - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    @staticmethod
    def iou_matrix(tracks, detections):
        """Cost matrix = 1 - IoU."""
        n_tracks, n_dets = len(tracks), len(detections)
        matrix = np.ones((n_tracks, n_dets))
        for i, track in enumerate(tracks):
            bbox_t = track.get_bbox()
            for j, det in enumerate(detections):
                matrix[i, j] = 1.0 - ByteTrack.iou(bbox_t, det[:4])
        return matrix

    def associate(self, tracks, detections, iou_threshold):
        """
        Hungarian matching between tracks and detections.
        Returns (matches, unmatched_tracks, unmatched_dets).
        """
        if len(tracks) == 0 or len(detections) == 0:
            return [], list(range(len(tracks))), list(range(len(detections)))

        cost_matrix = self.iou_matrix(tracks, detections)
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        matches = []
        unmatched_tracks = set(range(len(tracks)))
        unmatched_dets = set(range(len(detections)))

        for row, col in zip(row_indices, col_indices):
            iou = 1.0 - cost_matrix[row, col]
            if iou >= iou_threshold:
                matches.append((row, col))
                unmatched_tracks.discard(row)
                unmatched_dets.discard(col)

        return matches, list(unmatched_tracks), list(unmatched_dets)

    def update(self, detections):
        """
        Process a new frame of detections.

        detections: list of [x, y, w, h, confidence]

        Returns: list of [x, y, w, h, track_id, confidence]
        """
        # Split into high-score and low-score detections
        high_dets = [d for d in detections if d[4] >= self.track_thresh]
        low_dets = [d for d in detections if self.low_thresh <= d[4] < self.track_thresh]

        # Predict all tracks
        for track in self.tracks:
            track.predict()

        # Stage 1: associate high-score detections with all tracks
        matches_1, u_track_1, u_det_1 = self.associate(
            self.tracks, high_dets, self.match_thresh
        )

        for t_idx, d_idx in matches_1:
            self.tracks[t_idx].update(high_dets[d_idx][:4], high_dets[d_idx][4])

        # Stage 2: associate low-score detections with unmatched tracks
        if len(low_dets) > 0 and len(u_track_1) > 0:
            u_tracks = [self.tracks[i] for i in u_track_1]
            matches_2, u_track_2, _ = self.associate(u_tracks, low_dets, 0.5)

            for t_idx, d_idx in matches_2:
                u_tracks[t_idx].update(low_dets[d_idx][:4], low_dets[d_idx][4])

        # Create new tracks from unmatched high-score detections
        for d_idx in u_det_1:
            det = high_dets[d_idx]
            self.tracks.append(Track(self.next_id, det[:4], det[4]))
            self.next_id += 1

        # Remove dead tracks
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        # Return active tracks
        results = []
        for track in self.tracks:
            if track.hits >= self.min_hits:
                bbox = track.get_bbox()
                results.append([
                    bbox[0], bbox[1], bbox[2], bbox[3],
                    track.track_id, track.confidence
                ])

        return results
