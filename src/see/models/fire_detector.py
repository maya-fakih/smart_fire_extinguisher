"""
Fire Detector Module - YOLO Inference Analysis.

Analyzes YOLO inference results that come from the IMX500 camera chip.
The YOLO model runs on-device (on the camera hardware), so this module
just parses, filters, and clusters the detections into meaningful structures.

Classes:
    FireDetector: Main analyzer - parses YOLO metadata and clusters detections
"""

# this part is for fire detector, we analyze what we get from our YOLO trained Model
# YOLO runs on-chip on the IMX500 camera — we just parse the results from metadata
import logging
import numpy as np
from picamera2.devices.imx500 import IMX500
from shapely.geometry import box as shapely_box
from shapely.ops import unary_union
from see.models.vision_model_base import VisionModel
from see.models.detection import Detection
from see.models.fire_cluster import FireCluster

logger = logging.getLogger(__name__)

# Diagnostic throttle: detect() runs ~30x/sec. Log the shape/count picture at
# INFO every Nth call so the feed isn't silent but the log isn't flooded.
# Set to 1 temporarily for every-frame detail; 30 is roughly once per second.
_DIAG_EVERY = 30


# ── FireDetector ──────────────────────────────────────────────────────────────
# analyzes YOLO results that come from the IMX500 camera chip
# YOLO runs ON the camera hardware — FireDetector just parses and analyzes results
# inherits from VisionModel so it is forced to implement load()
class FireDetector(VisionModel):
    """
    Analyzes YOLO fire/smoke detection results from IMX500 camera.

    The YOLO model runs on the IMX500 chip (on-device inference).
    FireDetector receives the raw detection outputs and:
    1. Filters by confidence threshold
    2. Separates fire from smoke from other classes
    3. Merges overlapping boxes of the same class
    4. Clusters connected boxes into fire groups
    5. Computes spatial and area statistics

    Attributes:
        _imx500: IMX500 device object (owned by camera.py)
        _conf_threshold: Minimum confidence to accept detection
        _labels: Dict mapping class IDs to class names
    """

    # ── Init ──────────────────────────────────────────────────────────────────
    # receives conf_threshold and labels from VisionFuser (who reads config)
    # imx500 object is passed in from camera.py — FireDetector does not own hardware
    def __init__(self, imx500: IMX500, conf_threshold: float, labels: dict, picam2=None):
        """
        Initialize FireDetector with configuration.

        Args:
            imx500: IMX500 device object (owned by camera)
            conf_threshold: Minimum confidence for detections (0.0-1.0)
            labels: Dict mapping class ID to class name
        """
        self._imx500        = imx500            # IMX500 object owned by camera.py
        self._picam2        = picam2            # Picamera2 — needed for convert_inference_coords
        self._conf_threshold = conf_threshold   # minimum confidence to accept a box
        self._labels        = labels            # class id → label name from config
                                                # example: {0: "fire", 1: "other", 2: "smoke"}
        self._diag_calls    = 0                 # detect() call counter for throttled logging

    # ── Load ──────────────────────────────────────────────────────────────────
    # nothing to load here — model is loaded by camera.py onto the IMX500 chip
    # we implement load() because VisionModel forces us to, but it does nothing
    def load(self) -> None:
        """
        Load model (no-op for FireDetector).

        The YOLO model is loaded by camera.py onto the IMX500 chip.
        This method exists only to satisfy VisionModel interface contract.
        """
        pass                                    # model loading happens in camera.py

    # ── Detect ────────────────────────────────────────────────────────────────
    # main method — parses IMX500 metadata, runs full analysis
    # returns (clusters, raw_detections) for VisionFuser to build VisionSnapshot
    def detect(self, metadata, frame_width: int, frame_height: int) -> tuple[list[FireCluster], list[Detection]]:
        """
        Main detection pipeline: parse YOLO → filter → merge → cluster.

        This is the primary entry point. It orchestrates the full pipeline:
        1. Extract YOLO results from IMX500 metadata
        2. Filter by confidence threshold
        3. Merge overlapping boxes of same class
        4. Cluster connected boxes (connected components)
        5. Compute spatial and area statistics

        Args:
            metadata: IMX500 metadata dict from camera capture
            frame_width: Width of frame in pixels
            frame_height: Height of frame in pixels

        Returns:
            tuple[list[FireCluster], list[Detection]]:
                - clusters: List of FireCluster objects (may be empty)
                - raw_detections: All filtered detections (unmerged)
        """

        # total pixel count of the frame — needed for area ratio calculations
        frame_area = frame_width * frame_height

        # ── Step 1: Get YOLO results from IMX500 metadata ─────────────────────
        # YOLO already ran on-chip — we just extract the output tensors
        # get_outputs() returns the raw inference results from the camera
        outputs = self._imx500.get_outputs(metadata)

        # Throttled diagnostics — tells us what the chip is actually returning.
        self._diag_calls += 1
        diag = (self._diag_calls % _DIAG_EVERY == 1)

        # if camera returned nothing (no detections or not ready) → return empty
        if outputs is None:
            if diag:
                logger.info(
                    "FireDetector: get_outputs() returned None "
                    "(no inference result this frame — model not producing output, "
                    "or no detections). conf_threshold=%s", self._conf_threshold
                )
            return [], []

        if diag:
            try:
                shape_info = [
                    (np.asarray(o).shape if o is not None else None) for o in outputs
                ]
            except Exception:
                shape_info = "uninspectable"
            logger.info(
                "FireDetector: get_outputs() OK | n_tensors=%s | shapes=%s",
                len(outputs), shape_info
            )

        # IMX500 YOLO output format (verified from raw tensor dump):
        #   outputs[0] shape=(300, 4)  — boxes as [x1, y1, x2, y2] in pixels (corner format)
        #   outputs[1] shape=(300,)    — confidence scores [0, 1]
        #   outputs[2] shape=(300,)    — class ids
        #   outputs[3] shape=(1,)      — number of VALID detections (rest is zero-padding)
        # Always slice to n_valid so we never process padding rows.
        n_valid = int(np.asarray(outputs[3]).ravel()[0])
        boxes   = np.asarray(outputs[0])[:n_valid]   # shape (n_valid, 4)
        scores  = np.asarray(outputs[1])[:n_valid]   # shape (n_valid,)
        classes = np.asarray(outputs[2])[:n_valid]   # shape (n_valid,)

        raw_detections = []
        fire_boxes     = []
        smoke_boxes    = []

        # Diagnostic survival counters.
        _n_in = len(boxes)
        _dropped_conf = 0
        _dropped_label = 0

        for box, score, class_id in zip(boxes, scores, classes):

            # skip low confidence detections
            if score < self._conf_threshold:
                _dropped_conf += 1
                continue

            # skip "other" class — not relevant for fire analysis
            class_id = int(class_id)
            if class_id not in self._labels or self._labels[class_id] == "other":
                _dropped_label += 1
                continue

            # box is [x1, y1, x2, y2] in pixels (corner format from IMX500)
            # convert to [cx, cy, w, h] which the rest of the pipeline expects
            x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])

            # ── Coordinate-space fix ─────────────────────────────────────────
            # The IMX500 returns boxes in the NETWORK's input space (e.g. 320 or
            # 640 px depending on the .rpk), NOT in the captured-frame space.
            # Feeding those straight through made cluster origin_x reach 1.10 and
            # crash system_state's [0,1] guard. convert_inference_coords() maps a
            # network-space [x0,y0,w,h] box to true frame pixels using metadata.
            # Wrapped in try/except: if the API shape differs on this picamera2
            # version, we fall back to the raw values rather than killing the loop.
            try:
                if self._picam2 is not None:
                    conv = self._imx500.convert_inference_coords(
                        (x1, y1, x2 - x1, y2 - y1), metadata, self._picam2
                    )
                    # convert_inference_coords returns (x, y, w, h) rect in frame px
                    cx_px, cy_px, cw_px, ch_px = conv
                    x1, y1 = float(cx_px), float(cy_px)
                    x2, y2 = x1 + float(cw_px), y1 + float(ch_px)
            except Exception as e:
                if diag:
                    logger.warning(
                        "FireDetector: convert_inference_coords failed (%s) — "
                        "using raw coords; boxes/aim may be off until fixed",
                        type(e).__name__
                    )

            w = x2 - x1
            h = y2 - y1
            x = x1 + w / 2   # center x
            y = y1 + h / 2   # center y

            # area_ratio = how much of the frame this box covers
            area_ratio = (w * h) / frame_area

            # wrap into clean Detection object
            detection = Detection(
                label=self._labels[class_id],           # "fire" or "smoke" from labels dict
                confidence=float(score),
                bbox=(int(x), int(y), int(w), int(h)),
                area_ratio=area_ratio
            )

            raw_detections.append(detection)            # save for VisionSnapshot

            if self._labels[class_id] == "fire":
                fire_boxes.append(detection)
            else:
                smoke_boxes.append(detection)

        if diag:
            # If raw_in > 0 but survivors == 0, the filters are eating everything:
            #   dropped_conf high  → conf_threshold too high OR scores are 0-1 vs 0-100 scale mismatch
            #   dropped_label high → class ids from chip don't match labels.json keys
            # If raw_in == 0 → the chip produced an empty box array (model/threshold on-chip).
            sample = None
            if _n_in > 0:
                # raw first row, BEFORE any filtering — reveals coord scale (0-1 vs pixels)
                # and score scale (0-1 vs 0-100). If box values are < 1.0 the coords are
                # normalized and need imx500.convert_inference_coords() — boxes would
                # otherwise draw as 1px dots at the origin and look invisible.
                try:
                    sample = {
                        "box0": [float(v) for v in np.atleast_1d(boxes[0]).ravel()[:4]],
                        "score0": float(np.atleast_1d(scores[0]).ravel()[0]),
                        "class0": int(np.atleast_1d(classes[0]).ravel()[0]),
                    }
                except Exception as e:
                    sample = f"sample-failed: {e}"
            logger.info(
                "FireDetector: filter | raw_in=%s survived=%s "
                "(dropped_conf=%s dropped_label=%s) | fire=%s smoke=%s | conf_thr=%s | sample=%s",
                _n_in, len(raw_detections), _dropped_conf, _dropped_label,
                len(fire_boxes), len(smoke_boxes), self._conf_threshold, sample
            )

        # ── Step 4: Merge same-class boxes that intersect ─────────────────────
        # two fire boxes touching → become one bigger fire box
        # two smoke boxes touching → become one bigger smoke box
        # non-overlapping boxes stay separate
        fire_boxes  = self._merge_boxes(fire_boxes,  frame_area)
        smoke_boxes = self._merge_boxes(smoke_boxes, frame_area)

        # ── Step 5 & 6: Build clusters + compute area ratios ──────────────────
        # groups fire and smoke boxes into FireCluster objects
        # area ratios are computed inside _build_clusters
        # pass width + height so origin_x/y can be normalized to [0, 1]
        clusters = self._build_clusters(fire_boxes, smoke_boxes, frame_width, frame_height)

        # ── Step 7: Sort + Return ─────────────────────────────────────────────
        # sort clusters by total_area_ratio descending — largest fire first
        # this ensures VisionFuser and ThinkEngine always see the biggest threat first
        clusters.sort(key=lambda c: c.total_area_ratio, reverse=True)

        # clusters → VisionFuser uses to build VisionSnapshot
        # raw_detections → VisionSnapshot needs them as is
        return clusters, raw_detections


    # ── Helper: check if two boxes intersect ──────────────────────────────────
    # returns True if they overlap, False if they don't
    # used by _merge_boxes and _build_clusters
    def _boxes_intersect(self, a: Detection, b: Detection) -> bool:
        """
        Check if two bounding boxes intersect.

        Uses separating axis theorem: boxes overlap if they overlap
        on both X and Y axes simultaneously.

        Args:
            a: First Detection object
            b: Second Detection object

        Returns:
            bool: True if boxes overlap, False otherwise
        """

        # convert center format (x,y,w,h) to edges
        a_left,  a_right  = a.bbox[0] - a.bbox[2]/2,  a.bbox[0] + a.bbox[2]/2
        a_top,   a_bottom = a.bbox[1] - a.bbox[3]/2,  a.bbox[1] + a.bbox[3]/2
        b_left,  b_right  = b.bbox[0] - b.bbox[2]/2,  b.bbox[0] + b.bbox[2]/2
        b_top,   b_bottom = b.bbox[1] - b.bbox[3]/2,  b.bbox[1] + b.bbox[3]/2

        # overlap horizontally: A's left < B's right AND A's right > B's left
        horizontal_overlap = a_left < b_right and a_right > b_left
        # overlap vertically: A's top < B's bottom AND A's bottom > B's top
        vertical_overlap   = a_top  < b_bottom and a_bottom > b_top

        # BOTH must be true for real intersection
        return horizontal_overlap and vertical_overlap


    # ── Helper: merge two boxes into one bigger box that contains both ────────
    # takes highest confidence, keeps same label, recomputes area_ratio
    def _merge_two_boxes(self, a: Detection, b: Detection, frame_area: int) -> Detection:
        """
        Merge two overlapping boxes into one larger box.

        Creates the minimal bounding box that contains both input boxes.
        Inherits the label from the input (assumes same label).
        Takes the higher confidence score.
        Recomputes area_ratio based on new size.

        Args:
            a: First Detection object
            b: Second Detection object
            frame_area: Total frame area in pixels (for area_ratio computation)

        Returns:
            Detection: Merged bounding box
        """

        # get edges of both boxes
        a_left,  a_right  = a.bbox[0] - a.bbox[2]/2,  a.bbox[0] + a.bbox[2]/2
        a_top,   a_bottom = a.bbox[1] - a.bbox[3]/2,  a.bbox[1] + a.bbox[3]/2
        b_left,  b_right  = b.bbox[0] - b.bbox[2]/2,  b.bbox[0] + b.bbox[2]/2
        b_top,   b_bottom = b.bbox[1] - b.bbox[3]/2,  b.bbox[1] + b.bbox[3]/2

        # union box = take most extreme edge from each side
        new_left   = min(a_left,   b_left)
        new_right  = max(a_right,  b_right)
        new_top    = min(a_top,    b_top)
        new_bottom = max(a_bottom, b_bottom)

        # convert edges back to center format (x,y,w,h)
        new_w = new_right  - new_left
        new_h = new_bottom - new_top
        new_x = new_left   + new_w / 2
        new_y = new_top    + new_h / 2

        return Detection(
            label=a.label,
            confidence=max(a.confidence, b.confidence),
            bbox=(int(new_x), int(new_y), int(new_w), int(new_h)),
            area_ratio=(new_w * new_h) / frame_area
        )


    # ── Helper: merge all intersecting boxes in a list ────────────────────────
    # keeps looping until no intersecting pairs remain
    def _merge_boxes(self, boxes: list[Detection], frame_area: int) -> list[Detection]:
        """
        Merge all overlapping boxes in a list into non-overlapping groups.

        Iteratively merges pairs of intersecting boxes until no overlaps remain.
        Boxes of the same class that touch get merged into larger boxes.

        Args:
            boxes: List of Detection objects (all same class)
            frame_area: Total frame area in pixels

        Returns:
            list[Detection]: Reduced list of non-overlapping boxes
        """

        merged = True
        while merged:
            merged = False
            result = []
            used   = [False] * len(boxes)

            for i in range(len(boxes)):
                if used[i]:
                    continue

                current = boxes[i]

                for j in range(i + 1, len(boxes)):
                    if used[j]:
                        continue
                    if self._boxes_intersect(current, boxes[j]):
                        current  = self._merge_two_boxes(current, boxes[j], frame_area)
                        used[j]  = True
                        merged   = True

                result.append(current)

            boxes = result

        return boxes


    # ── Helper: compute union area in pixels for a list of non-overlapping boxes
    def _compute_area_pixels(self, boxes: list[Detection]) -> float:
        """
        Compute total area covered by a list of non-overlapping boxes.

        Simple sum of individual box areas (safe since boxes don't overlap).
        Returned as pixels, not normalized.

        Args:
            boxes: List of Detection objects (assumed non-overlapping)

        Returns:
            float: Total area in pixels
        """
        total = 0.0
        for box in boxes:
            total += box.bbox[2] * box.bbox[3]      # w × h
        return total


    # ── Helper: compute union area in pixels for mixed boxes (fire + smoke) ───
    # subtracts overlapping regions between every unique pair to avoid double counting
    def compute_union_area_pixels(self, all_boxes: list[Detection]) -> float:
        """
        Compute the union area (in pixels) of a set of axis-aligned bounding
        boxes. Each pixel is counted at most once, regardless of how many
        boxes overlap or whether one box is fully contained in another.

        Uses Shapely's unary_union (plane-sweep polygon union via GEOS).
        Correct for any n ≥ 0 — including nested boxes and 3+ box overlap
        at the same region — without inclusion-exclusion combinatorics.

        Args:
            all_boxes: List of Detection objects (may overlap or be empty)

        Returns:
            float: Union area in pixels (0.0 if input is empty)
        """
        if not all_boxes:
            return 0.0

        polygons = [
            shapely_box(
                b.bbox[0] - b.bbox[2] / 2,   # left
                b.bbox[1] - b.bbox[3] / 2,   # top
                b.bbox[0] + b.bbox[2] / 2,   # right
                b.bbox[1] + b.bbox[3] / 2,   # bottom
            )
            for b in all_boxes
        ]
        return float(unary_union(polygons).area)


    # ── Helper: group merged fire and smoke boxes into FireCluster objects ────
    def _build_clusters(self, fire_boxes: list[Detection], smoke_boxes: list[Detection], frame_width: int, frame_height: int, cluster_id: int = 0) -> list[FireCluster]:
        """
        Group overlapping fire and smoke boxes into connected components (clusters).

        Uses connected components algorithm (BFS) where boxes are connected if they
        overlap. All directly or indirectly connected boxes form one cluster.

        Example: FireBox A touches SmokeBox B, SmokeBox B touches SmokeBox C
                 → All three form one cluster even if A and C don't touch directly

        For each cluster, computes:
        - Composition (has_fire, has_smoke)
        - Center point (normalized [0, 1] — average of box centers ÷ frame dims)
        - Area statistics (union, fire-only, smoke-only)
        - Confidence metrics (average, primary)

        Args:
            fire_boxes: List of fire Detection objects (already merged)
            smoke_boxes: List of smoke Detection objects (already merged)
            frame_width: Frame width in pixels (used to normalize origin_x)
            frame_height: Frame height in pixels (used to normalize origin_y)
            cluster_id: Starting cluster ID (default 0)

        Returns:
            list[FireCluster]: List of FireCluster objects (may be empty)
        """

        # ── Connected Components Clustering ───────────────────────────────────
        # treat ALL boxes (fire + smoke) as nodes in a graph
        # two boxes are connected if they intersect
        # a cluster = a group of boxes all connected to each other (directly or through chain)
        #
        # example:
        #   FA touches SB, SB touches SC → FA + SB + SC = one cluster
        #   even if FA and SC never touch directly

        frame_area = frame_width * frame_height     # needed for area ratios below
        all_boxes  = fire_boxes + smoke_boxes       # all boxes together in one list
        n          = len(all_boxes)

        if n == 0:
            return []

        # ── Step 1: Build adjacency — which boxes touch which ─────────────────
        # adjacency[i] = list of box indices that box i intersects with
        adjacency = [[] for _ in range(n)]

        for i in range(n):
            for j in range(i + 1, n):
                if self._boxes_intersect(all_boxes[i], all_boxes[j]):
                    adjacency[i].append(j)          # i touches j
                    adjacency[j].append(i)          # j touches i (both directions)

        # ── Step 2: BFS to find connected groups ──────────────────────────────
        # visited[i] = True means box i already belongs to a cluster
        visited  = [False] * n
        clusters = []

        for start in range(n):
            if visited[start]:
                continue                            # already in a cluster, skip

            # start a new cluster from this box
            # BFS queue starts with this box
            queue        = [start]
            visited[start] = True
            group        = []                       # all box indices in this cluster

            while queue:
                current = queue.pop(0)             # take first item from queue (BFS)
                group.append(current)

                for neighbor in adjacency[current]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)     # add neighbor to explore next

            # ── Step 3: Build FireCluster from this group ─────────────────────
            group_boxes   = [all_boxes[i] for i in group]
            group_fire    = [b for b in group_boxes if b.label == "fire"]
            group_smoke   = [b for b in group_boxes if b.label == "smoke"]

            # average center of all boxes = cluster origin (in pixels)
            # then normalize to [0, 1] — hardware-agnostic, model-portable
            # (0.5, 0.5) = image center, which is what ACT expects as feedback signal
            origin_x_px = sum(b.bbox[0] for b in group_boxes) / len(group_boxes)
            origin_y_px = sum(b.bbox[1] for b in group_boxes) / len(group_boxes)
            origin_x    = origin_x_px / frame_width
            origin_y    = origin_y_px / frame_height

            # Backstop: clamp to [0, 1] so a residual coord-space mismatch can
            # never crash the capture loop on system_state's [0,1] guard again.
            # With convert_inference_coords working, this is a no-op; if conversion
            # is unavailable/imperfect the box may be slightly off but the loop
            # survives and still draws/aims. A value hitting 0 or 1 exactly is the
            # signal that conversion still needs attention.
            origin_x = min(1.0, max(0.0, origin_x))
            origin_y = min(1.0, max(0.0, origin_y))

            # average confidence across all boxes in cluster
            # more reliable than max — represents overall cluster certainty
            primary_confidence = sum(b.confidence for b in group_boxes) / len(group_boxes)

            # primary box = still the highest confidence box (for primary_bbox and primary_label)
            primary_box = max(group_boxes, key=lambda b: b.confidence)

            # composite label based on what's in this cluster
            has_fire  = len(group_fire)  > 0
            has_smoke = len(group_smoke) > 0

            if has_fire and has_smoke:
                composite_label = "fire_smoke"
            elif has_fire:
                composite_label = "fire"
            else:
                composite_label = "smoke"

            # compute area ratios
            fire_area_pixels  = self._compute_area_pixels(group_fire)
            smoke_area_pixels = self._compute_area_pixels(group_smoke)
            total_area_pixels = self.compute_union_area_pixels(group_boxes)

            clusters.append(FireCluster(
                cluster_id         = cluster_id,
                has_fire           = has_fire,
                has_smoke          = has_smoke,
                composite_label    = composite_label,
                fire_boxes         = group_fire,
                smoke_boxes        = group_smoke,
                box_count          = len(group_boxes),
                origin_x           = origin_x,
                origin_y           = origin_y,
                primary_bbox       = primary_box.bbox,
                total_area_ratio   = total_area_pixels / frame_area,
                fire_area_ratio    = fire_area_pixels  / frame_area,
                smoke_area_ratio   = smoke_area_pixels / frame_area,
                primary_label      = primary_box.label,
                primary_confidence = primary_box.confidence,
            ))
            cluster_id += 1

        return clusters