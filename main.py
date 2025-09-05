#!/usr/bin/env python3

import logging
import os

from config import FLASK_HOST, FLASK_PORT, validate_config
from routes import get_flask_app

# Set up logging
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level),
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    # Validate configuration before starting
    validate_config()

    # Get the Flask app from routes module
    app = get_flask_app()

    # Run the Flask application
    logger.info(f"Starting server on {FLASK_HOST}:{FLASK_PORT}")
    app.run(host=FLASK_HOST, port=FLASK_PORT)


if __name__ == "__main__":
    main()
