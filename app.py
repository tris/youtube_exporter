#!/usr/bin/env python3

import logging
import os

from dotenv import load_dotenv

from config import validate_config
from routes import get_flask_app

# Load environment variables from .env file
load_dotenv()

# Set up logging
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level),
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Validate configuration
validate_config()

# Create Flask app instance for Flask CLI
app = get_flask_app()

if __name__ == "__main__":
    from config import FLASK_HOST, FLASK_PORT

    logger.info(f"Starting server on {FLASK_HOST}:{FLASK_PORT}")
    app.run(host=FLASK_HOST, port=FLASK_PORT)
