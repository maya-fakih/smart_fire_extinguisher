#this part is for fire detector, we analyze what we get from our YOLO trained Model
from dataclasses import dataclass
from vision_model_base import VisionModel
from ultralytics import YOLO


# ── Data Classes ──
# these are just blueprints/containers for the data we will work with
# no logic here, just structure

#container for one single YOLO detection — one bounding box
@dataclass
class Detection:
    label: str                         # "fire" or "smoke"
    confidence: float                  # how sure YOLO is, between 0 and 1
    bbox: tuple[int, int, int, int]    # (x, y, w, h) in pixels
    area_ratio: float                  # box area ÷ frame area

# container for one fire cluster — a group of connected fire/smoke boxes
@dataclass
class FireCluster:
    cluster_id: int                    # just a number to identify this cluster
    has_fire: bool                     # does this cluster have any fire boxes?
    has_smoke: bool                    # does this cluster have any smoke boxes?
    fire_boxes: list[Detection]        # all fire detections in this cluster
    smoke_boxes: list[Detection]       # all smoke detections in this cluster
    union_area: float                  # total area of cluster (no double counting)
    primary_confidence: float          # highest confidence box in this cluster
    composite_label: str               # "fire_smoke", "fire", "smoke"


# ── FireDetector ──
# this is the main class — it owns the YOLO model and runs all the analysis
# inherits from VisionModel so it is forced to implement load()
class FireDetector(VisionModel):
    
    # ── Init ──
    # receives model path and confidence threshold from VisionFuser (who reads config)
    # does NOT open config.json itself
    def __init__(self, model_path: str, conf_threshold: float):
        self._model_path = model_path        # where the YOLO model file lives
        self._conf_threshold = conf_threshold # minimum confidence to accept a box
        self._model = None                    # YOLO model, empty until load() is called
    
    # ── Load ──
    # loads the YOLO model from disk into memory
    # called once at startup by VisionFuser before any detection happens
    def load(self) -> None:
        self._model = YOLO(self._model_path)                                 
    
    # ── Detect ──
    # main method — takes one frame, runs full analysis, returns clusters + raw detections
    # this is what VisionFuser calls every time it captures a frame
    def detect(self, frame, frame_width: int, frame_height: int) -> tuple[list[FireCluster], list[Detection]]:
        
        # total pixel count of the frame — we need this to calculate area ratios later
        frame_area = frame_width * frame_height
        
        # ── Step 1: Run YOLO on the frame ──
        # self._model is the YOLO model we loaded in load()
        # conf= tells YOLO "ignore any detection below this confidence"
        # results is a list, [0] gives us the first (and only) frame's results
        results = self._model(frame, conf=self._conf_threshold)
        
        # ── Steps 2 & 3: Parse YOLO output → separate into fire and smoke lists ──
        # we skip class_id == 1 ("other") — only useful for YOLO training, not for us
        # raw_detections keeps ALL boxes because VisionSnapshot needs them later
        fire_boxes = []         # only fire detections go here
        smoke_boxes = []        # only smoke detections go here
        raw_detections = []     # everything (fire + smoke) goes here unmodified
        
        for box in results[0].boxes:
            
            # box.cls is the class YOLO assigned — 0=fire, 1=other, 2=smoke
            # it comes as a tensor so int() converts it to a normal Python number
            class_id = int(box.cls[0])
            
            if class_id == 1:              # skip "other" — we dont analyze it
                continue
            
            # box.xywh gives [x_center, y_center, width, height] in pixels
            # [0] because ultralytics wraps it in an extra list layer
            x, y, w, h = box.xywh[0]
            
            # box.conf is how sure YOLO is — comes as tensor, float() makes it a number
            confidence = float(box.conf[0])
            
            # area_ratio = what percentage of the frame this box covers
            # example: box is 100x50 pixels, frame is 640x480 (307200 pixels)
            # area_ratio = 5000 / 307200 = 0.016 → box covers 1.6% of the frame
            area_ratio = (float(w) * float(h)) / frame_area
            
            # wrap everything into a clean Detection object
            detection = Detection(
                label="fire" if class_id == 0 else "smoke",  # 0→fire, 2→smoke
                confidence=confidence,
                bbox=(int(x), int(y), int(w), int(h)),
                area_ratio=area_ratio
            )
            
            raw_detections.append(detection)   # save unmodified for VisionSnapshot
            
            if class_id == 0:
                fire_boxes.append(detection)
            else:
                smoke_boxes.append(detection)
        
        # ── Step 4: Merge same-class boxes that intersect ──
        # two fire boxes touching → become one bigger fire box
        # two smoke boxes touching → become one bigger smoke box
        # non-overlapping boxes stay separate
        fire_boxes  = self._merge_boxes(fire_boxes,  frame_area)
        smoke_boxes = self._merge_boxes(smoke_boxes, frame_area)
        
        # ── Step 5: Build clusters ──
        # group fire and smoke boxes that are near each other into FireCluster objects
        clusters = self._build_clusters(fire_boxes, smoke_boxes)
        
        # ── Step 6: Compute union area for each cluster ──
        # fill in the union_area field we left as 0.0 in _build_clusters
        for cluster in clusters:
            cluster.union_area = self._compute_union_area(cluster)
        
        # ── Step 7: Return ──
        # clusters → VisionFuser uses to build VisionSnapshot
        # raw_detections → VisionSnapshot needs them as is
        return clusters, raw_detections


    # ── Helper: check if two boxes intersect ──
    # returns True if they overlap, False if they dont
    # used by _merge_boxes and _build_clusters
    def _boxes_intersect(self, a: Detection, b: Detection) -> bool:
        
        # convert center format (x,y,w,h) to edges
        # left = center_x - half_width, right = center_x + half_width
        a_left,  a_right  = a.bbox[0] - a.bbox[2]/2,  a.bbox[0] + a.bbox[2]/2
        a_top,   a_bottom = a.bbox[1] - a.bbox[3]/2,  a.bbox[1] + a.bbox[3]/2
        b_left,  b_right  = b.bbox[0] - b.bbox[2]/2,  b.bbox[0] + b.bbox[2]/2
        b_top,   b_bottom = b.bbox[1] - b.bbox[3]/2,  b.bbox[1] + b.bbox[3]/2
        
        # overlap horizontally: A's left is before B's right AND A's right is after B's left
        horizontal_overlap = a_left < b_right and a_right > b_left
        # overlap vertically: A's top is above B's bottom AND A's bottom is below B's top
        vertical_overlap   = a_top  < b_bottom and a_bottom > b_top
        
        # BOTH must be true for real intersection
        return horizontal_overlap and vertical_overlap


    # ── Helper: merge two boxes into one bigger box that contains both ──
    # takes highest confidence, same label, recomputes area_ratio
    def _merge_two_boxes(self, a: Detection, b: Detection, frame_area: int) -> Detection:
        
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
            label=a.label,                              # same class, label doesnt change
            confidence=max(a.confidence, b.confidence), # keep higher confidence
            bbox=(int(new_x), int(new_y), int(new_w), int(new_h)),
            area_ratio=(new_w * new_h) / frame_area
        )


    # ── Helper: merge all intersecting boxes in a list ──
    # keeps looping until no intersecting pairs remain
    def _merge_boxes(self, boxes: list[Detection], frame_area: int) -> list[Detection]:
        
        merged = True
        while merged:
            merged = False                         # assume nothing to merge this pass
            result = []
            used = [False] * len(boxes)            # used[i]=True means box i was absorbed
            
            for i in range(len(boxes)):
                if used[i]:                        # already absorbed, skip
                    continue
                
                current = boxes[i]
                
                for j in range(i+1, len(boxes)):
                    if used[j]:
                        continue
                    if self._boxes_intersect(current, boxes[j]):
                        current = self._merge_two_boxes(current, boxes[j], frame_area)
                        used[j] = True
                        merged = True              # found a merge → do another pass
                
                result.append(current)
            
            boxes = result
        
        return boxes


    # ── Helper: group merged fire and smoke boxes into FireCluster objects ──
    # fire box + nearby smoke box → fire_smoke cluster
    # fire box alone → fire cluster
    # smoke box alone → smoke cluster
    def _build_clusters(self, fire_boxes: list[Detection], smoke_boxes: list[Detection], cluster_id: int = 0) -> list[FireCluster]:
        
        clusters  = []
        used_fire  = [False] * len(fire_boxes)
        used_smoke = [False] * len(smoke_boxes)
        
        # loop through fire boxes — check if any smoke box touches each one
        for i, f_box in enumerate(fire_boxes):
            matched_smoke = []
            
            for j, s_box in enumerate(smoke_boxes):
                if used_smoke[j]:
                    continue
                if self._boxes_intersect(f_box, s_box):
                    matched_smoke.append(s_box)
                    used_smoke[j] = True
            
            used_fire[i] = True
            
            cluster = FireCluster(
                cluster_id=cluster_id,
                has_fire=True,
                has_smoke=len(matched_smoke) > 0,
                fire_boxes=[f_box],
                smoke_boxes=matched_smoke,
                union_area=0.0,                    # filled in after by _compute_union_area
                primary_confidence=f_box.confidence,
                composite_label="fire_smoke" if matched_smoke else "fire"
            )
            clusters.append(cluster)
            cluster_id += 1
        
        # leftover smoke boxes that didnt match any fire → smoke only clusters
        for j, s_box in enumerate(smoke_boxes):
            if used_smoke[j]:
                continue
            cluster = FireCluster(
                cluster_id=cluster_id,
                has_fire=False,
                has_smoke=True,
                fire_boxes=[],
                smoke_boxes=[s_box],
                union_area=0.0,
                primary_confidence=s_box.confidence,
                composite_label="smoke"
            )
            clusters.append(cluster)
            cluster_id += 1
        
        return clusters


    # ── Helper: compute total area of a cluster without double counting overlaps ──
    # adds all box areas, then subtracts any overlapping regions between pairs
    def _compute_union_area(self, cluster: FireCluster) -> float:
        
        all_boxes = cluster.fire_boxes + cluster.smoke_boxes
        
        if not all_boxes:
            return 0.0
        
        # sum all box areas
        total_area = 0.0
        for box in all_boxes:
            total_area += box.bbox[2] * box.bbox[3]   # w × h
        
        # subtract overlapping regions between every unique pair
        for i in range(len(all_boxes)):
            for j in range(i + 1, len(all_boxes)):
                a = all_boxes[i]
                b = all_boxes[j]
                
                a_left   = a.bbox[0] - a.bbox[2] / 2
                a_right  = a.bbox[0] + a.bbox[2] / 2
                a_top    = a.bbox[1] - a.bbox[3] / 2
                a_bottom = a.bbox[1] + a.bbox[3] / 2
                b_left   = b.bbox[0] - b.bbox[2] / 2
                b_right  = b.bbox[0] + b.bbox[2] / 2
                b_top    = b.bbox[1] - b.bbox[3] / 2
                b_bottom = b.bbox[1] + b.bbox[3] / 2
                
                inter_w = min(a_right, b_right) - max(a_left, b_left)
                inter_h = min(a_bottom, b_bottom) - max(a_top, b_top)
                
                if inter_w > 0 and inter_h > 0:
                    total_area -= inter_w * inter_h
        
        return total_area
