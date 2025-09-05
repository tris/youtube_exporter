#!/usr/bin/env python3

import logging
from config import FLASK_HOST, FLASK_PORT, validate_config
from routes import get_flask_app

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    # Validate configuration before starting
    validate_config()
        
    # Get the Flask app from routes module
    app = get_flask_app()
    
    # Run the Flask application
    logger.info(f"Starting server on {FLASK_HOST}:{FLASK_PORT}")
    app.run(host=FLASK_HOST, port=FLASK_PORT)

if __name__ == '__main__':
    main()