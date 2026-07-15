from __future__ import annotations

from typing import Any

from chatkit.server import StreamingResult
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from chatkit_app.server import MedicalDiagnosisChatKitServer
from chatkit_app.store import InMemoryChatKitStore


app = FastAPI(title="Medical Skill Hub ChatKit API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["*"],
)

store = InMemoryChatKitStore()
server = MedicalDiagnosisChatKitServer(store=store)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chatkit")
async def chatkit_endpoint(request: Request) -> Response:
    context: dict[str, Any] = {}
    result = await server.process(await request.body(), context=context)
    if isinstance(result, StreamingResult):
        return StreamingResponse(result, media_type="text/event-stream")
    return Response(content=result.json, media_type="application/json")
