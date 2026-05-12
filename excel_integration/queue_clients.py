from __future__ import annotations

import json
import time
import uuid
import inspect
import importlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from .config import IntegrationSettings


class QueueClient(ABC):
    @abstractmethod
    def pull_request(self) -> dict[str, Any] | None:
        raise NotImplementedError

    @abstractmethod
    def push_result(self, result: dict[str, Any]) -> None:
        raise NotImplementedError


class FileQueueClient(QueueClient):
    def __init__(self, requests_dir: Path, results_dir: Path):
        self.requests_dir = requests_dir
        self.results_dir = results_dir
        self.requests_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def pull_request(self) -> dict[str, Any] | None:
        request_files = sorted(self.requests_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
        if not request_files:
            return None

        request_file = request_files[0]
        with request_file.open("rt", encoding="utf-8") as f:
            payload = json.load(f)

        request_file.unlink(missing_ok=True)
        return payload

    def push_result(self, result: dict[str, Any]) -> None:
        request_id = result.get("request_id") or str(uuid.uuid4())
        target = self.results_dir / f"{request_id}.json"
        with target.open("wt", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, separators=(",", ":"), default=str)

    def enqueue_request(self, payload: dict[str, Any], request_id: str | None = None) -> str:
        rid = request_id or payload.get("request_id") or str(uuid.uuid4())
        payload = dict(payload)
        payload["request_id"] = rid
        payload.setdefault("created_utc", int(time.time()))
        target = self.requests_dir / f"{rid}.json"
        with target.open("wt", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, separators=(",", ":"), default=str)
        return rid


class SolaceQueueClient(QueueClient):
    def __init__(self, settings: IntegrationSettings):
        self.settings = settings
        self._messaging_service = None
        self._receiver = None
        self._publisher = None

    @staticmethod
    def _first_available(obj: Any, method_names: list[str]) -> Any:
        for name in method_names:
            if hasattr(obj, name):
                return getattr(obj, name)
        raise AttributeError(f"None of the methods are available: {method_names}")

    @staticmethod
    def _call_compat(method: Any, /, **kwargs: Any) -> Any:
        signature = inspect.signature(method)
        accepted = {k: v for k, v in kwargs.items() if k in signature.parameters}
        return method(**accepted)

    @staticmethod
    def _decode_inbound_message(message: Any) -> str:
        for method_name in ("get_payload_as_string", "get_payload_as_bytes", "get_payload"):
            if hasattr(message, method_name):
                payload = getattr(message, method_name)()
                if isinstance(payload, bytes):
                    return payload.decode("utf-8")
                return str(payload)
        return str(message)

    def _validate_settings(self) -> None:
        required = {
            "RF_SOLACE_HOST": self.settings.solace_host,
            "RF_SOLACE_VPN": self.settings.solace_vpn,
            "RF_SOLACE_USERNAME": self.settings.solace_username,
            "RF_SOLACE_PASSWORD": self.settings.solace_password,
        }
        missing = [key for key, value in required.items() if not str(value or "").strip()]
        if missing:
            raise ValueError(f"Missing Solace environment settings: {', '.join(missing)}")

    def _ensure_connected(self) -> None:
        if self._messaging_service is not None and self._receiver is not None and self._publisher is not None:
            return

        self._validate_settings()

        try:
            messaging_service_module = importlib.import_module("solace.messaging.messaging_service")
            queue_module = importlib.import_module("solace.messaging.resources.queue")
            topic_module = importlib.import_module("solace.messaging.resources.topic")
            MessagingService = getattr(messaging_service_module, "MessagingService")
            Queue = getattr(queue_module, "Queue")
            Topic = getattr(topic_module, "Topic")
        except Exception as exc:
            raise RuntimeError(
                "Solace Python API not available. Install with 'pip install solace-pubsubplus'."
            ) from exc

        properties = {
            "solace.messaging.transport.host": self.settings.solace_host,
            "solace.messaging.service.vpn-name": self.settings.solace_vpn,
            "solace.messaging.authentication.scheme.basic.username": self.settings.solace_username,
            "solace.messaging.authentication.scheme.basic.password": self.settings.solace_password,
        }

        builder = MessagingService.builder().from_properties(properties)
        messaging_service = builder.build()
        self._first_available(messaging_service, ["connect", "connect_async"])()

        receiver_builder = self._first_available(
            messaging_service,
            ["create_persistent_message_receiver_builder", "persistent_message_receiver_builder"],
        )()

        if hasattr(receiver_builder, "with_message_auto_acknowledgement"):
            receiver_builder = receiver_builder.with_message_auto_acknowledgement()
        elif hasattr(receiver_builder, "with_auto_acknowledgement"):
            receiver_builder = receiver_builder.with_auto_acknowledgement()

        request_queue = Queue.durable_exclusive_queue(self.settings.solace_request_queue)
        receiver_build_method = self._first_available(receiver_builder, ["build"])
        receiver = receiver_build_method(request_queue)
        self._first_available(receiver, ["start", "start_async"])()

        publisher_builder = self._first_available(
            messaging_service,
            ["create_persistent_message_publisher_builder", "persistent_message_publisher_builder"],
        )()
        publisher = self._first_available(publisher_builder, ["build"])()
        self._first_available(publisher, ["start", "start_async"])()

        self._messaging_service = messaging_service
        self._receiver = receiver
        self._publisher = publisher
        self._topic_type = Topic

    def pull_request(self) -> dict[str, Any] | None:
        self._ensure_connected()

        receive_method = self._first_available(
            self._receiver,
            ["receive_message", "receive", "receive_inbound_message"],
        )

        message = self._call_compat(
            receive_method,
            timeout=1000,
            timeout_millis=1000,
            timeout_ms=1000,
        )

        if message is None:
            return None

        payload = self._decode_inbound_message(message)
        data = json.loads(payload)

        if "request_id" not in data:
            data["request_id"] = str(uuid.uuid4())

        return data

    def push_result(self, result: dict[str, Any]) -> None:
        self._ensure_connected()

        payload = json.dumps(result, ensure_ascii=False, separators=(",", ":"), default=str)
        outbound_builder = self._first_available(
            self._messaging_service,
            ["message_builder", "create_message_builder"],
        )()

        if hasattr(outbound_builder, "with_string_payload"):
            outbound_builder = outbound_builder.with_string_payload(payload)
            message = outbound_builder.build()
        elif hasattr(outbound_builder, "with_payload"):
            outbound_builder = outbound_builder.with_payload(payload.encode("utf-8"))
            message = outbound_builder.build()
        else:
            message = outbound_builder.build(payload)

        destination = self._topic_type.of(self.settings.solace_result_topic)
        publish_method = self._first_available(self._publisher, ["publish", "publish_await_response", "send"])
        self._call_compat(
            publish_method,
            message=message,
            outbound_message=message,
            destination=destination,
            topic=destination,
        )


def build_queue_client(settings: IntegrationSettings) -> QueueClient:
    if settings.queue_backend == "solace":
        return SolaceQueueClient(settings)

    return FileQueueClient(settings.file_queue_requests_dir, settings.file_queue_results_dir)
