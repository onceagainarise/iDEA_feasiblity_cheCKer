import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    """Application settings"""
    
    # Groq Configuration (using GROQ_API_KEY and GROQ_MODEL from .env)
    groq_api_key: str = Field(..., env="GROQ_API_KEY")
    groq_model: str = Field(..., env="GROQ_MODEL")
    
    # ArXiv API settings
    arxiv_base_url: str = "http://export.arxiv.org/api/query"
    arxiv_max_results: int = 20
    
    # News API settings
    news_api_key: Optional[str] = Field(None, env="NEWS_API_KEY")
    news_api_url: str = "https://newsapi.org/v2"
    
    # Patent search settings
    patent_api_key: Optional[str] = Field(None, env="PATENT_API_KEY")
    
    # Agent settings
    max_research_iterations: int = 3
    confidence_threshold: float = 0.7
    
    # Logging
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Global settings instance
settings = Settings()
