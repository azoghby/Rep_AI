from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class SignalReading:
    host_time_ms: int
    signal_value: float
    raw_sample: str = ""
    device_time_ms: int | None = None


class SignalSource(Protocol):
    def connect(self) -> None:
        ...

    def disconnect(self) -> None:
        ...

    def read(self) -> SignalReading | None:
        ...

    def read_many(
        self,
        max_samples: int = 250,
        max_duration_seconds: float = 0.05,
    ) -> list[SignalReading]:
        ...
