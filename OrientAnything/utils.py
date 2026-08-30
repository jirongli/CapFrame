import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict

@dataclass
class OrientResult:
    azimuth: float
    polar: float
    rotation: float
    confidence: float

    @classmethod
    def from_dict(cls, orient_dict: Dict) -> 'OrientResult':
        return cls(azimuth=orient_dict['azimuth'],
                   polar=orient_dict['polar'],
                   rotation=orient_dict['rotation'],
                   confidence=orient_dict['confidence'],
                   )