"""FastAPI server application serving Strawberry GraphQL and REST health endpoints."""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from strawberry.fastapi import GraphQLRouter

from mdk_trading_oracle.api.schema import schema

logger = logging.getLogger("mdk_api")

graphql_app = GraphQLRouter(schema)

app = FastAPI(
    title="MDK Trading Oracle GraphQL API",
    description="GraphQL API for BIST institutional order flows, candlesticks, and broker PnL.",
    version="1.0.0",
)

# Enable CORS for local React development server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount GraphQL router
app.include_router(graphql_app, prefix="/graphql")


@app.get("/")
@app.get("/health")
def health_check():
    """Healthcheck endpoint for monitoring."""
    return {
        "status": "healthy",
        "service": "mdk-trading-oracle-api",
        "graphql_endpoint": "/graphql",
    }
