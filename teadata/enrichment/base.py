from abc import ABC, abstractmethod
from typing import Any, Dict


class Enricher(ABC):
    """Each Enricher loads a dataset via config and writes attributes onto District/Campus."""

    @abstractmethod
    def apply(self, repo, cfg_path: str, year: int) -> Dict[str, Any]: ...
