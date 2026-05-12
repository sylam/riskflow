from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class IntegrationSettings:
    queue_backend: str
    file_queue_requests_dir: Path
    file_queue_results_dir: Path
    solace_host: str
    solace_vpn: str
    solace_username: str
    solace_password: str
    solace_request_queue: str
    solace_result_topic: str


def load_settings() -> IntegrationSettings:
    root = Path(os.getenv("RF_FILE_QUEUE_ROOT", ".rf_queue")).resolve()
    return IntegrationSettings(
        queue_backend=os.getenv("RF_QUEUE_BACKEND", "file").strip().lower(),
        file_queue_requests_dir=Path(os.getenv("RF_FILE_QUEUE_REQUESTS_DIR", str(root / "requests"))).resolve(),
        file_queue_results_dir=Path(os.getenv("RF_FILE_QUEUE_RESULTS_DIR", str(root / "results"))).resolve(),
        solace_host=os.getenv("RF_SOLACE_HOST", ""),
        solace_vpn=os.getenv("RF_SOLACE_VPN", ""),
        solace_username=os.getenv("RF_SOLACE_USERNAME", ""),
        solace_password=os.getenv("RF_SOLACE_PASSWORD", ""),
        solace_request_queue=os.getenv("RF_SOLACE_REQUEST_QUEUE", "riskflow/requests"),
        solace_result_topic=os.getenv("RF_SOLACE_RESULT_TOPIC", "riskflow/results"),
    )
