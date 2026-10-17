import logging

from fastapi import FastAPI

from app.api.routes import health, playground

logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="crusoe-hack",
    description="Shared Brain: setup checks for Crusoe and Modal",
    version="0.1.0",
)

app.include_router(health.router)
app.include_router(playground.router)
