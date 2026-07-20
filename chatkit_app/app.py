from __future__ import annotations

from typing import Any

from chatkit.server import StreamingResult
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from chatkit_app.server import MedicalDiagnosisChatKitServer
from chatkit_app.store import InMemoryChatKitStore
from chatkit_app.translation import DisplayTranslator, normalize_display_language


app = FastAPI(title="Medical Skill Hub ChatKit API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:43179", "http://127.0.0.1:43179"],
    allow_credentials=True,
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["*"],
)

translator = DisplayTranslator()
store = InMemoryChatKitStore(translator=translator)
server = MedicalDiagnosisChatKitServer(store=store, translator=translator)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chatkit")
async def chatkit_endpoint(request: Request) -> Response:
    context: dict[str, Any] = {
        "display_language": normalize_display_language(
            request.headers.get("X-Display-Language")
        )
    }
    result = await server.process(await request.body(), context=context)
    if isinstance(result, StreamingResult):
        return StreamingResponse(result, media_type="text/event-stream")
    return Response(content=result.json, media_type="application/json")
