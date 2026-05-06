1. sense snapshot and vision are done
2. it doesnt make any sense to have the actions written in the think snapshot
the think snapshots only purpose is to align and have ready the sensor and vison snapshots aligned with the closest time stamp

the think snapshot is the call to save to the database?
or should we have something else align and write to database? annd the think snapshot will only store the derived values that should be fed as a vestor into the think engine?

maybe it is more optimal to write to db already and only have the think snapshot as a caches quick acess to the ferived values w we will do its function as a yeild one snapshot after the other for optimality eh? even cache it like we do in python for like lets say fibonsacci


for now this is what i will keep in the think snapshot 
import exceptions
from dataclasses import dataclass, feild
from datetime import datetime
from sense import SensorSnapshot
from see import VisionSnapshot

@dataclass
class ThinkSnapshot():
    # system variables
    timestamp: datetime

    # system snapshots
    sensor_snapshot: SensorSnapshot
    vision_snapshot: VisionSnapshot

    # think derived
    growth_rate: float      # derived through weighted calculation of the rate of all inputs
    escelation_trend: str   # labeled by xgboost
    danger_level: int
    danger_label: str       # may be redundant if danger levels are enums
    recommended_action: str # based on danger level and available hardware

    # all vector points needed to be inputted to xgboost
    # they are derived from db walkback of all saved sensor snapshots and vision snapshots saved
    # calculations are never saved in the db bcz its faster to recompute, we can max cache them for speed
    growth_rates: dict      # key = sensor name, value = velocity or growth rate
    growth_trend: dict      # key = sensor name, value = acceleration or growth trend


okay now we have think_logs these are the calculations? 
we need to think again about how we will really structure the think so it works... but for now this is what we have im not sure how to continue

