from pydantic_settings import BaseSettings, SettingsConfigDict  # type: ignore


class Settings(BaseSettings):
    # Define the settings here
    app_name: str = "Single Agent DLP"
    AUTHOR_NAME: str
    AUTHOR_EMAIL: str

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
