import os

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    PROJECT_NAME: str = "Iris - Image Recognition Service"
    PROJECT_DESCRIPTION: str = "Image recognition microservice for Sleeved"
    PROJECT_VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"

    DATABASE_URL: str = os.getenv(
        "DATABASE_URL", "mysql+pymysql://user:password@mysql:3306/fastapi"
    )

    class Config:
        env_file = ".env"


settings = Settings()
