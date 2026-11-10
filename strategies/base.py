from abc import ABC, abstractmethod
from core.types import Entity


class Strategy(ABC):
    @abstractmethod
    def replace(self, entity: Entity, mapper) -> str:
        """Return the text that should replace `entity`."""
