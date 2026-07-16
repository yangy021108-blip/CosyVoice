"""Command-line entry point: python -m api_server.main."""

from __future__ import annotations

import logging

import uvicorn

from api_server.app import create_app
from api_server.config import Settings


app = create_app()


def main() -> None:
    settings = Settings.from_env()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    uvicorn.run(
        "api_server.main:app",
        host=settings.host,
        port=settings.port,
        workers=1,
    )


if __name__ == "__main__":
    main()
