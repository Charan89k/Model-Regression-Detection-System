from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from mrds.core.exceptions.base import MRDSError
from mrds.presentation.api.routers import analysis, health, runs

app = FastAPI(
    title="Model Regression Detection System (MRDS)",
    description="API for triggering and analyzing LLM evaluation runs.",
    version="0.1.0",
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Exception Handling
@app.exception_handler(MRDSError)
async def mrds_exception_handler(request: Request, exc: MRDSError):
    """Translates internal domain errors to HTTP 400 Responses."""
    return JSONResponse(
        status_code=400,
        content={"error": exc.__class__.__name__, "message": str(exc)},
    )

# Register Routers
app.include_router(health.router)
app.include_router(runs.router)
app.include_router(analysis.router)
