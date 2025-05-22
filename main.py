import logging
import time

from fastapi import FastAPI

from app.core.config import settings
from app.db.session import init_db
from app.router.routes import include_routes

app = FastAPI(
    title=settings.PROJECT_NAME,
    description=settings.PROJECT_DESCRIPTION,
    version=settings.PROJECT_VERSION,
)

# Include all routes
include_routes(app)


@app.on_event("startup")
async def startup_event():
    # Try to connect to the database with retries
    max_retries = 10
    for attempt in range(max_retries):
        try:
            init_db()
            logging.info("Database initialized successfully")
            break
        except Exception as e:
            logging.error(
                "Database connection failed (attempt {}/{}): {}".format(
                    attempt + 1, max_retries, str(e)
                )
            )
            if attempt < max_retries - 1:
                wait_time = 5  # seconds
                logging.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logging.warning("Max retries reached. Starting without database.")
