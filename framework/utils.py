"""
    utils.py
    --------
    Defines utility functions for the framework
"""

from typing import Dict, Any, Union
import yaml
import io
import base64

KEY_START_ANGlE: str = "angle_start"
KEY_END_ANGLE: str = "angle_end"
KEY_ROTATION_INTERVAL: str = "rotation_interval"
KEY_OBJECT_DIMENSION: str = "obj_dim"
KEY_PADDING_WIDTH: str = "padding"

class Config:

    _cfg: Dict[str, Any]
    _scope: str

    def __init__(self, filepath: str = "config.yaml", scope: str = "data_gen") -> None:
        with open(file="config.yaml", mode="r") as stream:
            self._cfg = yaml.safe_load(stream=stream)
        self._scope = "data_gen"

    def __getitem__(self, key: str) -> Union[Any, None]:
        return self._cfg[self._scope].get(key, None)

if __name__ == "__main__":
    pass