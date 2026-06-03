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
import numpy as np
from picamera2.devices.imx500 import IMX500
from shapely.geometry import box as shapely_box
from shapely.ops import unary_union
from see.models.vision_model_base import VisionModel
from see.models.detection import Detection
from see.models.fire_cluster import FireCluster


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
    def __init__(self, imx500: IMX500, conf_threshold: float, labels: dict):
        """
        Initialize FireDetector with configuration.

        Args:
            imx500: IMX500 device object (owned by camera)
            conf_threshold: Minimum confidence for detections (0.0-1.0)
            labels: Dict mapping class ID to class name
        """
        self._imx500        = imx500            # IMX500 object owned by camera.py
        self._conf_threshold = conf_threshold   # minimum confidence to accept a box
        self._labels        = labels            # class id → label name from config
                                                # example: {0: "fire", 1: "other", 2: "smoke"}

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

        # if camera returned nothing (no detections or not ready) → return empty
        if outputs is None:
            return [], []

        # IMX500 YOLO output format: boxes, scores, classes
        # boxes  → array of [x, y, w, h] for each detection
        # scores → confidence value for each detection
        # classes → class id for each detection
        boxes, scores, classes = outputs[0][0], outputs[1][0], outputs[2][0]

        # ── Steps 2 & 3: Parse output → separate into fire and smoke ──────────
        # skip class_id == 1 ("other") — only useful for YOLO training, not for us
        # raw_detections keeps ALL boxes because VisionSnapshot needs them later

        # guard against single-detection squeeze — IMX500 returns scalar
        # numpy.float32 instead of array when there is exactly one detection
        boxes   = np.atleast_1d(boxes)
        scores  = np.atleast_1d(scores)
        classes = np.atleast_1d(classes)

        raw_detections = []
        fire_boxes     = []
        smoke_boxes    = []

        for box, score, class_id in zip(boxes, scores, classes):

            # skip low confidence detections
            if score < self._conf_threshold:
                continue

            # skip "other" class — not relevant for fire analysis
            class_id = int(class_id)
            if class_id not in self._labels or self._labels[class_id] == "other":
                continue

            # box comes as [x_center, y_center, width, height] in pixels
            x, y, w, h = box[0], box[1], box[2], box[3]

            # area_ratio = how much of the frame this box covers
            # example: box 100x50 on 640x480 frame → 5000/307200 = 0.016 = 1.6%
            area_ratio = (float(w) * float(h)) / frame_area

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