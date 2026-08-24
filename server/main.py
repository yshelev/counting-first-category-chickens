from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app = FastAPI()

birds = []
stats = {"total": 0, "grade1": 0, "brak": 0}
subscribers = set()

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def index():
    return FileResponse("static/index.html")

@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    subscribers.add(websocket)

    await websocket.send_json({"type": "init", "birds": birds, "stats": stats})
    try:
        while True:
            await websocket.receive_text()  
    except WebSocketDisconnect:
        subscribers.remove(websocket)

@app.post("/data/bird")
async def receive_bird(bird: dict):
    """Прием данных от процессора"""
    birds.append(bird)
    stats["total"] = len(birds)

    grade1 = sum(1 for b in birds if b.get("grade", "1") == "1")
    stats["grade1"] = grade1
    stats["brak"] = stats["total"] - grade1

    msg = {"type": "bird", "bird": bird, "stats": stats}
    for ws in subscribers.copy():
        try:
            await ws.send_json(msg)
        except:
            subscribers.discard(ws)

    return {"ok": True}

@app.get("/stats")
async def get_stats():
    return {"birds": birds, "stats": stats}

@app.get("/health")
async def health():
    return {"ok": True, "birds": len(birds), "subscribers": len(subscribers)}