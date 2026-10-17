"""Agentic AI for Deepfake Audio Detection — model service.

Standalone FastAPI backend. The Lovable dashboard talks to it through the
`Inference endpoint` and `Training endpoint` fields on the Settings page.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .routers import dataset, health, inference, training

app = FastAPI(
    title="Deepfake Audio Detection API",
    description=(
        "Log-mel + MFCC feature extraction, a spectrogram CNN with a prosody "
        "fusion branch, SHAP/LIME explainability and background training jobs."
    ),
    version="1.0.1",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_pna_and_cors_headers(request, call_next):
    if request.method == "OPTIONS":
        from fastapi import Response
        response = Response(status_code=200)
    else:
        response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "*"
    response.headers["Access-Control-Allow-Private-Network"] = "true"
    return response

app.include_router(health.router)
app.include_router(inference.router)
app.include_router(training.router)
app.include_router(dataset.router)


@app.get("/", tags=["health"])
async def root() -> dict:
    return {
        "service": "deepfake-audio-detection",
        "docs": "/docs",
        "endpoints": ["/health", "/predict", "/train", "/status/{job_id}", "/dataset/stats"],
    }


def run() -> None:  # `python -m app.main`
    import uvicorn

    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=True)


if __name__ == "__main__":
    run()
