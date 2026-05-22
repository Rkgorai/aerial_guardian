import cv2
import numpy as np


class GMC:
    """Global Motion Compensation (GMC) for camera motion compensation.
    
    Uses ORB keypoints and homography matching with RANSAC to align bounding boxes
    across consecutive video frames, compensating for rapid drone ego-motion.
    """

    def __init__(self, method="orb", max_features=1000):
        self.method = method.lower()
        self.max_features = max_features
        
        # Internal state
        self.prev_frame = None
        self.prev_kps = None
        self.prev_descs = None
        
        # Initialize ORB detector
        if self.method == "orb":
            self.detector = cv2.ORB_create(nfeatures=max_features)
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        else:
            self.detector = None
            self.matcher = None

    def reset(self):
        """Reset internal frame cache."""
        self.prev_frame = None
        self.prev_kps = None
        self.prev_descs = None

    def apply(self, frame, detections=None):
        """Compute the camera motion matrix (homography H) between the previous and current frame.
        
        frame: BGR frame (numpy array)
        detections: list of [x, y, w, h, ...] bounding boxes to mask out
        Returns: 3x3 Homography matrix mapping prev_frame coords to current_frame coords.
        """
        # If method is 'none', return identity matrix
        if self.method == "none" or self.detector is None or frame is None:
            return np.eye(3, dtype=np.float32)

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Create mask to exclude foreground objects (detections) from background keypoint estimation
        mask = np.ones_like(gray, dtype=np.uint8) * 255
        if detections is not None:
            for det in detections:
                x, y, w, h = det[:4]
                # Convert center-format bbox to corner bounds
                x1 = max(0, int(x - w / 2))
                y1 = max(0, int(y - h / 2))
                x2 = min(frame.shape[1], int(x + w / 2))
                y2 = min(frame.shape[0], int(y + h / 2))
                if x2 > x1 and y2 > y1:
                    mask[y1:y2, x1:x2] = 0

        # Detect keypoints and compute descriptors
        kps, descs = self.detector.detectAndCompute(gray, mask=mask)

        # If it's the first frame, cache and return identity
        if self.prev_frame is None or self.prev_descs is None or descs is None or len(descs) < 10:
            self.prev_frame = gray
            self.prev_kps = kps
            self.prev_descs = descs
            return np.eye(3, dtype=np.float32)

        # Match descriptors with KNN (k=2)
        knn_matches = self.matcher.knnMatch(self.prev_descs, descs, 2)
        
        # Apply Lowe's ratio test (ratio threshold = 0.9)
        good_matches = []
        for match_pair in knn_matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.9 * n.distance:
                    good_matches.append(m)

        if len(good_matches) < 4:
            # Not enough matches, fallback to identity
            self.prev_frame = gray
            self.prev_kps = kps
            self.prev_descs = descs
            return np.eye(3, dtype=np.float32)

        # Extract coordinates of matched points
        src_pts = np.float32([self.prev_kps[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kps[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Find homography using RANSAC
        H, inliers = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # If homography estimation failed, fallback to identity
        if H is None:
            H = np.eye(3, dtype=np.float32)
        else:
            # Sanity check on H to prevent tracking explosions on erratic/random keypoint matches.
            # Real camera ego-motion should be relatively smooth between consecutive frames.
            tx = H[0, 2]
            ty = H[1, 2]
            det = H[0, 0] * H[1, 1] - H[0, 1] * H[1, 0]
            
            # If translation is extremely large, or scale/rotation is severely distorted (negative or too large/small),
            # fall back to identity mapping.
            if abs(tx) > 200 or abs(ty) > 200 or det < 0.6 or det > 1.6:
                H = np.eye(3, dtype=np.float32)

        # Update cache
        self.prev_frame = gray
        self.prev_kps = kps
        self.prev_descs = descs

        return H
