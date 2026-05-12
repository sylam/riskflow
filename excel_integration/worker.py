from __future__ import annotations

import argparse
import json
import time

from .config import load_settings
from .pricing_service import price_job
from .queue_clients import build_queue_client


def process_once(verbose: bool = False) -> bool:
    settings = load_settings()
    client = build_queue_client(settings)
    request = client.pull_request()
    if request is None:
        return False

    result = price_job(request)
    client.push_result(result)

    if verbose:
        print(json.dumps({
            "request_id": result.get("request_id"),
            "status": result.get("status"),
            "elapsed_ms": result.get("elapsed_ms"),
        }))

    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="RiskFlow Excel queue worker")
    parser.add_argument("--once", action="store_true", help="Process a single request and exit")
    parser.add_argument("--poll-seconds", type=float, default=1.0, help="Polling interval when running continuously")
    parser.add_argument("--verbose", action="store_true", help="Print status lines")
    args = parser.parse_args()

    if args.once:
        processed = process_once(verbose=args.verbose)
        if not processed and args.verbose:
            print('{"status":"idle"}')
        return

    while True:
        processed = process_once(verbose=args.verbose)
        if not processed:
            time.sleep(max(args.poll_seconds, 0.1))


if __name__ == "__main__":
    main()
