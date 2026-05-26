import numpy as np
from collections import deque
from scipy.spatial.distance import cdist

from aerial_guardian.tracking.algorithms.basetrack import TrackState
from aerial_guardian.tracking.algorithms.byte_tracker import STrack, BYTETracker, KalmanFilterXYWH
from aerial_guardian.tracking.algorithms.gmc import GMC
from aerial_guardian.tracking.algorithms.reid import ReIDManager


class BOTrack(STrack):
    """Represent a single track segment for BoT-SORT, adding appearance features and GMC."""

    def __init__(self, bbox, score, feature=None):
        super().__init__(bbox, score)
        self.features = deque(maxlen=100)
        self.alpha = 0.9
        self.smooth_feat = None
        self.curr_feature = None
        
        # Initialize appearance feature
        if feature is not None:
            self.update_features(feature)

    def update_features(self, feat):
        """Update features vector and smooth it using exponential moving average."""
        if feat is None:
            return
        feat = np.asarray(feat, dtype=np.float32)
        norm = np.linalg.norm(feat)
        if norm > 1e-5:
            feat = feat / norm
        self.curr_feature = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat.copy()
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
            smooth_norm = np.linalg.norm(self.smooth_feat)
            if smooth_norm > 1e-5:
                self.smooth_feat /= smooth_norm
        self.features.append(feat)

    @staticmethod
    def to_xywh(bbox):
        """Convert bounding box [x, y, w, h] to xywh representation (noop, but keeps API matching)."""
        return np.asarray(bbox, dtype=np.float32)

    def activate(self, kalman_filter, frame_id):
        """Activate a new track."""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.to_xywh(self._bbox))
        
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id
        self.time_since_update = 0

    def apply_gmc(self, H):
        """Compensate for camera ego-motion by applying homography matrix H.
        
        H: 3x3 Homography matrix mapping previous coordinates to current coordinates.
        This compensates position, velocity, and the covariance matrix.
        """
        if H is None or np.allclose(H, np.eye(3)):
            return

        R = H[:2, :2]
        R8x8 = np.kron(np.eye(4, dtype=np.float32), R)
        t = H[:2, 2]

        # Standard BoT-SORT matrix transformation
        self.mean = R8x8 @ self.mean
        self.mean[:2] += t
        self.covariance = R8x8 @ self.covariance @ R8x8.T

    def update(self, new_track, frame_id):
        """Update track with matching detection and append its appearance embedding."""
        self.frame_id = frame_id
        self.tracklet_len += 1
        
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.to_xywh(new_track._bbox)
        )
        self.score = new_track.score
        self.state = TrackState.Tracked
        self.is_activated = True
        self.time_since_update = 0
        self.history.append(new_track._bbox.copy())

        if new_track.curr_feature is not None:
            self.update_features(new_track.curr_feature)

    def re_activate(self, new_track, frame_id, new_id=False):
        """Re-activate lost track with matching detection and append appearance embedding."""
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.to_xywh(new_track._bbox)
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

        if new_track.curr_feature is not None:
            self.update_features(new_track.curr_feature)

    @property
    def bbox(self):
        """Get the current estimated bounding box [x_center, y_center, w, h]."""
        if self.mean is None:
            return self._bbox.copy()
        return self.mean[:4].copy()


class BOTSORT(BYTETracker):
    """BoT-SORT Tracker: Multi-Object Tracking with ReID and Global Motion Compensation (GMC)."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.kalman_filter = KalmanFilterXYWH()
        
        # Gating thresholds
        self.proximity_thresh = cfg.get("proximity_thresh", 0.5)
        self.appearance_thresh = cfg.get("appearance_thresh", 0.25)
        self.fuse_score = cfg.get("fuse_score", True)
        
        # GMC & ReID Managers
        gmc_method = cfg.get("gmc_method", "orb")
        self.gmc = GMC(method=gmc_method)
        
        reid_weights = cfg.get("reid_weights", "resnet18")
        device = cfg.get("device", "cpu")
        self.reid = ReIDManager(model_name=reid_weights, device=device)

    def get_fused_cost_matrix(self, tracks, detections):
        """Combine spatial (IoU) and appearance (ReID) distances to build a unified cost matrix.
        
        This aligns exactly with Ultralytics BoT-SORT distance calculation and gating.
        """
        if len(tracks) == 0 or len(detections) == 0:
            return np.empty((len(tracks), len(detections)), dtype=np.float32)

        # 1. Compute IoU distance (1 - IoU)
        dists = self.iou_distance(tracks, detections)
        
        # 2. Compute spatial proximity gating mask (True if spatial distance exceeds limit)
        dists_mask = dists > (1 - self.proximity_thresh)

        # 3. Fuse score if enabled (lower cost for high-confidence detections)
        if self.fuse_score:
            det_scores = np.array([det.score for det in detections], dtype=np.float32)
            det_scores = np.tile(det_scores, (len(tracks), 1))
            iou_sim = 1.0 - dists
            dists = 1.0 - iou_sim * det_scores

        # 4. Compute appearance / embedding distance using fast, robust cdist
        emb_dists = np.ones((len(tracks), len(detections)), dtype=np.float32)
        valid_track_idxs = [i for i, t in enumerate(tracks) if t.smooth_feat is not None]
        valid_det_idxs = [j for j, d in enumerate(detections) if d.curr_feature is not None]
        
        if len(valid_track_idxs) > 0 and len(valid_det_idxs) > 0:
            track_feats = np.array([tracks[i].smooth_feat for i in valid_track_idxs], dtype=np.float32)
            det_feats = np.array([detections[j].curr_feature for j in valid_det_idxs], dtype=np.float32)
            
            # Cosine distance computation
            raw_dists = np.clip(cdist(track_feats, det_feats, 'cosine'), 0.0, 2.0)
            
            # Map back to full cost grid
            for u, i in enumerate(valid_track_idxs):
                for v, j in enumerate(valid_det_idxs):
                    emb_dists[i, j] = raw_dists[u, v]

        # Rescale embedding distance to [0, 1] range (since raw cosine distance spans [0, 2])
        emb_dists /= 2.0
        
        # Gate and penalize poor appearance matches (if cosine distance exceeds appearance limit)
        emb_dists[emb_dists > (1 - self.appearance_thresh)] = 1.0
        
        # Gate and penalize poor spatial matches
        emb_dists[dists_mask] = 1.0
        
        # Final cost matrix takes the minimum of either spatial IoU or gated appearance
        cost_matrix = np.minimum(dists, emb_dists)
        return cost_matrix

    def associate_fused(self, tracks, detections, threshold):
        """Stage 1 Association using combined spatial and appearance distances."""
        if len(tracks) == 0 or len(detections) == 0:
            return [], list(range(len(tracks))), list(range(len(detections)))

        cost_matrix = self.get_fused_cost_matrix(tracks, detections)
        return self.linear_assignment(cost_matrix, threshold)

    def update(self, detections, frame=None):
        """Process a frame of detections with camera motion compensation and appearance extraction.
        
        detections: list of [x_center, y_center, w, h, confidence]
        frame: raw BGR frame image (numpy array)
        
        Returns: list of [x_center, y_center, w, h, track_id, confidence]
        """
        self.frame_id += 1
        
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        # 1. Global Motion Compensation (GMC)
        # Compute the homography mapping previous frame's space to the current frame
        H = self.gmc.apply(frame, detections)
        
        # 2. Apply GMC to existing active and lost tracks to update their Kalman Filter center coordinates
        for track in self.tracked_stracks:
            track.apply_gmc(H)
        for track in self.lost_stracks:
            track.apply_gmc(H)

        # 3. Extract appearance features (ReID embeddings) for all detections
        track_dets = []
        for det in detections:
            x, y, w, h, conf = det[:5]
            # Extract appearance feature embedding
            feat = self.reid.extract(frame, [x, y, w, h])
            track_det = BOTrack([x, y, w, h], conf, feature=feat)
            track_dets.append(track_det)

        # Split detections by confidence score
        high_dets = []
        low_dets = []
        for det in track_dets:
            if det.score >= self.track_thresh:
                high_dets.append(det)
            elif det.score >= self.low_thresh:
                low_dets.append(det)

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

        # Predict next state using Kalman Filter
        for track in strack_pool:
            track.predict()

        # Step 3: Stage 1 Association using combined spatial and appearance (ReID) distances
        matches_1, u_track_1, u_det_1 = self.associate_fused(
            strack_pool, high_dets, 0.8  # Fused matching threshold
        )

        for itracked, idet in matches_1:
            track = strack_pool[itracked]
            det = high_dets[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        # Step 4: Stage 2 Association (low-confidence detections with remaining active tracked tracks, pure spatial)
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

        # Step 5: Stage 3 Association (unconfirmed tracks with unmatched high-confidence detections, pure spatial)
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
