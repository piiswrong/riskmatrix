from dataclasses import dataclass
from datetime import datetime

@dataclass
class PnLEntry:
    timestamp: datetime
    value: float
