# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import atexit
import json
import os
import threading
from pathlib import Path
from typing import Any


_PROFILE_PATH_ENV = "VLLM_KV_OFFLOAD_PROFILE_PATH"


def kv_offload_profile_enabled() -> bool:
    return bool(os.getenv(_PROFILE_PATH_ENV))


class KVOffloadProfileWriter:
    """Append completed request profiles to a JSON array.

    This is intentionally enabled only by an env var so normal serving has no
    file IO overhead.
    """

    _instance: "KVOffloadProfileWriter | None" = None
    _instance_lock = threading.Lock()

    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._closed = False
        self._has_records = False
        self._fp = self.path.open("w", encoding="utf-8")
        self._fp.write("[\n")
        self._fp.flush()
        atexit.register(self.close)

    @classmethod
    def get(cls) -> "KVOffloadProfileWriter | None":
        path = os.getenv(_PROFILE_PATH_ENV)
        if not path:
            return None
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls(path)
            return cls._instance

    def write(self, record: dict[str, Any]) -> None:
        with self._lock:
            if self._closed:
                return
            if self._has_records:
                self._fp.write(",\n")
            json.dump(record, self._fp, ensure_ascii=False, sort_keys=True)
            self._has_records = True
            self._fp.flush()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._fp.write("\n]\n")
            self._fp.close()
            self._closed = True
