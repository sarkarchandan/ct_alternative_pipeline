"""
    utils.py
    --------
    Defines utility functions for the framework
"""

from typing import Dict, Any, Union
import yaml

KEY_START_ANGlE: str = "angle_start"
KEY_END_ANGLE: str = "angle_end"
KEY_ROTATION_INTERVAL: str = "rotation_interval"
KEY_OBJECT_DIMENSION: str = "obj_dim"
KEY_PADDING_WIDTH: str = "padding"

class Config:

    _cfg: Dict[str, Any]

    def __init__(self, 
        filepath: str = "config.yaml", 
        scope: str = "data_gen") -> None:
         with open(file="config.yaml", mode="r") as stream:
            cfg = yaml.safe_load(stream=stream)

    def __getitem__(self, key: str) -> Union[Any, None]:
        return self._cfg[self.scope].get(key, None)

    
    