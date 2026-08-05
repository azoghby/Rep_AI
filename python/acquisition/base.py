from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class SignalReading:
    host_time_ms: int
    signal_value: float


class SignalSource(Protocol):
    def connect(self) -> None:
        ...

    def disconnect(self) -> None:
        ...

    def read(self) -> SignalReading | None:
        ...
