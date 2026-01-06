import re
from typing import Any, Optional


def _parse_vehicle_num(v: Any) -> Optional[int]:
    if v is None:
        return None
    m = re.search(r"(\d+)", str(v))
    return int(m.group(1)) if m else None


__all__ = [
    "_parse_vehicle_num",
]
