import numpy as np
import scipy.linalg
from scipy.optimize import linear_sum_assignment
from collections import deque

from trackers.basetrack import BaseTrack, TrackState


class KalmanFilter:
    """Kalman filter for tracking aspect-ratio center bounding boxes.
    
    The 8-dimensional state space: [x, y, a, h, vx, vy, va, vh]
    where (x, y) is center, a is aspect ratio (w/h), h is height, and
    vx, vy, va, vh are their respective velocities.
    """

    def __init__(self):
        ndim, dt = 4, 1.0
        # Motion transition matrix F
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        
        # Observation matrix H
        self._update_mat = np.eye(ndim, 2 * ndim)
        
        # Uncertainty weights
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(self, measurement):
        """Initialize track state from a measurement."""
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3],
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        """Predict the next state distribution using constant velocity motion model."""
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3],
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = np.dot(mean, self._motion_mat.T)
        covariance = np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov
        return mean, covariance

    def project(self, mean, covariance):
        """Project state distribution to observation (measurement) space."""
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3],
        ]
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def update(self, mean, covariance, measurement):
        """Correct state distribution using measurement."""
        projected_mean, projected_cov = self.project(mean, covariance)

        chol_factor, lower = scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_mat.T).T, check_finite=False
        ).T
        innovation = measurement - projected_mean

        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance


class KalmanFilterXYWH(KalmanFilter):
    """Kalman filter for tracking center bounding boxes with width and height directly.
    
    The 8-dimensional state space: [x, y, w, h, vx, vy, vw, vh]
    where (x, y) is center, w is width, h is height, and
    vx, vy, vw, vh are their respective velocities.
    """

    def initiate(self, measurement):
        """Initialize track state from a measurement [x, y, w, h]."""
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[2],
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[2],
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[2],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[2],
            10 * self._std_weight_velocity * measurement[3],
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        """Predict the next state distribution using constant velocity motion model."""
        std_pos = [
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[2],
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[2],
            self._std_weight_velocity * mean[3],
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = np.dot(mean, self._motion_mat.T)
        covariance = np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov
        return mean, covariance

    def project(self, mean, covariance):
        """Project state distribution to observation (measurement) space."""
        std = [
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
        ]
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def multi_predict(self, mean, covariance):
        """Vectorized Kalman filter prediction step."""
        std_pos = [
            self._std_weight_position * mean[:, 2],
            self._std_weight_position * mean[:, 3],
            self._std_weight_position * mean[:, 2],
            self._std_weight_position * mean[:, 3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[:, 2],
            self._std_weight_velocity * mean[:, 3],
            self._std_weight_velocity * mean[:, 2],
            self._std_weight_velocity * mean[:, 3],
        ]
        sqr = np.square(np.r_[std_pos, std_vel]).T

        motion_cov = [np.diag(sqr[i]) for i in range(len(mean))]
        motion_cov = np.asarray(motion_cov)

        mean = np.dot(mean, self._motion_mat.T)
        left = np.dot(self._motion_mat, covariance).transpose((1, 0, 2))
        covariance = np.dot(left, self._motion_mat.T) + motion_cov

        return mean, covariance


class STrack(BaseTrack):
    """Represent a single track segment for ByteTrack."""

    def __init__(self, bbox, score):
        # bbox format: [x_center, y_center, w, h]
        self._bbox = np.asarray(bbox, dtype=np.float32)
        self.score = score
        self.tracklet_len = 0
        
        # State vector variables initialized to None until activated
        self.kalman_filter = None
        self.mean = None
        self.covariance = None
        self.is_activated = False
        
        # History
        self.history = deque(maxlen=30)
        self.history.append(self._bbox.copy())

    @staticmethod
    def to_xyah(bbox):
        """Convert center bbox [x, y, w, h] to aspect ratio representation [x, y, a, h]."""
        x, y, w, h = bbox
        a = w / h if h > 0 else 0.0
        return np.array([x, y, a, h], dtype=np.float32)

    def predict(self):
        """Advance the Kalman filter state by one frame."""
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0  # Zero out height velocity for non-tracked
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)
        self.time_since_update += 1

    def activate(self, kalman_filter, frame_id):
        """Activate a new track."""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.to_xyah(self._bbox))
        
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        # Confirmed immediately only on first frame of video
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id
        self.time_since_update = 0

    def re_activate(self, new_track, frame_id, new_id=False):
        """Re-activate a lost track."""
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.to_xyah(new_track._bbox)
        )
        self.score = new_track.score
        self.state = TrackState.Tracked
        self.is_activated = True
        self.time_since_update = 0
        self.frame_id = frame_id
        self.tracklet_len = 0
        if new_id:
            self.track_id = self.next_id()
        self.history.append(new_track._bbox.copy())

    def update(self, new_track, frame_id):
        """Update the track with a matching detection."""
        self.frame_id = frame_id
        self.tracklet_len += 1
        
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.to_xyah(new_track._bbox)
        )
        self.score = new_track.score
        self.state = TrackState.Tracked
        self.is_activated = True
        self.time_since_update = 0
        self.history.append(new_track._bbox.copy())

    @property
    def bbox(self):
        """Get the current estimated bounding box [x_center, y_center, w, h]."""
        if self.mean is None:
            return self._bbox.copy()
        x, y, a, h = self.mean[:4]
        w = a * h
        return np.array([x, y, w, h], dtype=np.float32)


class BYTETracker:
    """ByteTrack Multi-Object Tracker (Aligned with Ultralytics)."""

    def __init__(self, cfg):
        self.track_thresh = cfg.get("track_high_thresh", 0.25)
        self.low_thresh = cfg.get("track_low_thresh", 0.1)
        self.new_track_thresh = cfg.get("new_track_thresh", 0.25)
        self.match_thresh = cfg.get("match_thresh", 0.8)
        self.max_time_lost = cfg.get("track_buffer", 30)

        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.kalman_filter = KalmanFilter()
        STrack.reset_id()

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
    def iou_distance(tracks, detections):
        """Cost matrix = 1 - IoU."""
        if len(tracks) == 0 or len(detections) == 0:
            return np.empty((len(tracks), len(detections)), dtype=np.float32)
        
        cost_matrix = np.ones((len(tracks), len(detections)), dtype=np.float32)
        for i, track in enumerate(tracks):
            bbox_t = track.bbox
            for j, det in enumerate(detections):
                cost_matrix[i, j] = 1.0 - BYTETracker.iou(bbox_t, det.bbox)
        return cost_matrix

    def linear_assignment(self, cost_matrix, thresh):
        """Hungarian linear sum assignment matching."""
        if cost_matrix.size == 0:
            return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))

        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        matches = []
        unmatched_rows = set(range(cost_matrix.shape[0]))
        unmatched_cols = set(range(cost_matrix.shape[1]))

        for r, c in zip(row_indices, col_indices):
            cost = cost_matrix[r, c]
            if cost <= thresh:
                matches.append((r, c))
                unmatched_rows.discard(r)
                unmatched_cols.discard(c)

        return matches, list(unmatched_rows), list(unmatched_cols)

    def update(self, detections, frame=None):
        """Process a frame of detections.
        
        detections: list of [x_center, y_center, w, h, confidence]
        frame: raw image frame (ignored in ByteTrack, kept for api consistency)
        
        Returns: list of [x_center, y_center, w, h, track_id, confidence]
        """
        self.frame_id += 1
        
        # Output accumulation variables
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        # Split detections by confidence score
        high_dets = []
        low_dets = []
        
        for det in detections:
            x, y, w, h, conf = det[:5]
            track_det = STrack([x, y, w, h], conf)
            if conf >= self.track_thresh:
                high_dets.append(track_det)
            elif conf >= self.low_thresh:
                low_dets.append(track_det)

        # Separate activated (confirmed) vs unconfirmed tracks
        tracked_stracks = []
        unconfirmed = []
        
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        # Combine tracked and lost tracks to build matching pool
        strack_pool = self.joint_stracks(tracked_stracks, self.lost_stracks)

        # Predict current state for all tracks in pool
        for track in strack_pool:
            track.predict()

        # Step 3: First association (high-confidence detections with active/lost tracks)
        dists = self.iou_distance(strack_pool, high_dets)
        matches_1, u_track_1, u_det_1 = self.linear_assignment(dists, thresh=self.match_thresh)

        for itracked, idet in matches_1:
            track = strack_pool[itracked]
            det = high_dets[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        # Step 4: Second association (low-confidence detections with remaining active tracked tracks)
        r_tracked_stracks = [strack_pool[i] for i in u_track_1 if strack_pool[i].state == TrackState.Tracked]
        dists_low = self.iou_distance(r_tracked_stracks, low_dets)
        matches_2, u_track_2, _ = self.linear_assignment(dists_low, thresh=0.5)

        for itracked, idet in matches_2:
            track = r_tracked_stracks[itracked]
            det = low_dets[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        # Unmatched tracked tracks are transitioned to lost
        for it in u_track_2:
            track = r_tracked_stracks[it]
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        # Step 5: Third association (unconfirmed tracks with unmatched high-confidence detections)
        unconfirmed_dets = [high_dets[i] for i in u_det_1]
        dists_unconfirmed = self.iou_distance(unconfirmed, unconfirmed_dets)
        matches_3, u_unconfirmed, u_det_3 = self.linear_assignment(dists_unconfirmed, thresh=0.7)

        for itracked, idet in matches_3:
            unconfirmed[itracked].update(unconfirmed_dets[idet], self.frame_id)
            activated_stracks.append(unconfirmed[itracked])
            
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        # Step 6: Initiate new tracks from high-confidence unmatched detections
        for inew in u_det_3:
            track = unconfirmed_dets[inew]
            if track.score < self.new_track_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_stracks.append(track)

        # Step 7: Clean up lost tracks exceeding track_buffer length
        for track in self.lost_stracks:
            if self.frame_id - track.frame_id > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # Sync tracked, lost, and removed states
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, refind_stracks)
        
        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = self.sub_stracks(self.lost_stracks, removed_stracks)
        
        # Suppress duplicates
        self.tracked_stracks, self.lost_stracks = self.remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)

        # Return results (only for confirmed/activated tracks)
        results = []
        for track in self.tracked_stracks:
            if track.is_activated:
                bbox = track.bbox
                results.append([
                    bbox[0], bbox[1], bbox[2], bbox[3],
                    track.track_id, track.score
                ])

        return results

    @staticmethod
    def joint_stracks(tlista, tlistb):
        """Combine two lists of tracks while preventing ID duplicates."""
        exists = {}
        res = []
        for t in tlista:
            exists[t.track_id] = 1
            res.append(t)
        for t in tlistb:
            tid = t.track_id
            if not exists.get(tid, 0):
                exists[tid] = 1
                res.append(t)
        return res

    @staticmethod
    def sub_stracks(tlista, tlistb):
        """Subtract tracks in list B from list A based on track IDs."""
        track_ids_b = {t.track_id for t in tlistb}
        return [t for t in tlista if t.track_id not in track_ids_b]

    @staticmethod
    def remove_duplicate_stracks(stracksa, stracksb):
        """Suppress duplicated tracks by analyzing overlap distance."""
        pdist = BYTETracker.iou_distance(stracksa, stracksb)
        pairs = np.where(pdist < 0.15)
        dupa, dupb = [], []
        for p, q in zip(*pairs):
            timep = stracksa[p].frame_id - stracksa[p].start_frame
            timeq = stracksb[q].frame_id - stracksb[q].start_frame
            if timep > timeq:
                dupb.append(q)
            else:
                dupa.append(p)
        resa = [t for i, t in enumerate(stracksa) if i not in dupa]
        resb = [t for i, t in enumerate(stracksb) if i not in dupb]
        return resa, resb
