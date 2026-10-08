# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Test-support doubles for rigging's telemetry transport.

Shipped alongside the library (not under ``tests/``) so rigging's own suite and
downstream test suites import one copy rather than each carrying its own.
"""

import json
import threading
from collections.abc import Mapping
from typing import Any, cast


class _TelemetryResponse:
    status_code = 200

    def __init__(self, batch_id: str) -> None:
        self._batch_id = batch_id
        self.headers: dict[str, str] = {}

    def json(self) -> dict[str, object]:
        return {"batch_id": self._batch_id, "status": "accepted"}


class RecordingTelemetryTransport:
    """Capture telemetry records and return valid acknowledgements in tests."""

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []
        self.resources: list[dict[str, Any]] = []
        self.condition = threading.Condition()

    def post(self, endpoint: str, body: bytes, batch_id: str, timeout: tuple[float, float]) -> _TelemetryResponse:
        del endpoint, timeout
        payload = json.loads(body)
        with self.condition:
            self.records.extend(payload["records"])
            self.resources.append(payload["resource"])
            self.condition.notify_all()
        return _TelemetryResponse(batch_id)

    def close(self) -> None:
        pass

    def wait_for(self, count: int) -> list[dict[str, Any]]:
        with self.condition:
            assert self.condition.wait_for(lambda: len(self.records) >= count, timeout=5)
            return list(self.records)

    def record(self, name: str, attributes: Mapping[str, str]) -> dict[str, Any]:
        def matches(record: dict[str, Any]) -> bool:
            record_attributes = cast(dict[str, str], record["attributes"])
            return record["name"] == name and all(
                record_attributes.get(key) == value for key, value in attributes.items()
            )

        with self.condition:
            assert self.condition.wait_for(lambda: any(matches(record) for record in self.records), timeout=5)
            return [record for record in self.records if matches(record)][-1]

    def wait_for_value(self, name: str, attributes: Mapping[str, str], expected: float) -> None:
        with self.condition:
            assert self.condition.wait_for(
                lambda: any(
                    record["name"] == name
                    and record["value"] == expected
                    and all(
                        cast(dict[str, str], record["attributes"]).get(key) == value for key, value in attributes.items()
                    )
                    for record in self.records
                ),
                timeout=5,
            )
