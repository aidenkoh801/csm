from fastapi import FastAPI, WebSocket
from generator import load_csm_1b
import torchaudio
import numpy as np
import asyncio
from concurrent.futures import ThreadPoolExecutor

app = FastAPI()
executor = ThreadPoolExecutor()
# Load generator (consider caching in real applications)
generator = load_csm_1b(device="cuda")
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Receive parameters
            params = await websocket.receive_json()
            text = params.get("text", "Hello from Sesame.")
            speaker = params.get("speaker", 0)

            loop = asyncio.get_event_loop()
            audio_gen = generator.generate(
                text=text,
                speaker=speaker,
                context=[],
                max_audio_length_ms=10_000,
                temperature=0.9,
                topk=50,
                chunk_size=20
            )

            # Bridge synchronous generator to async
            def sync_generator():
                for chunk in audio_gen:
                    # Convert to numpy array and bytes
                    audio_np = chunk.cpu().numpy().astype(np.float32)
                    yield audio_np.tobytes()

            # Process chunks and send
            gen = await loop.run_in_executor(executor, sync_generator)
            for chunk_bytes in gen:
                await websocket.send_bytes(chunk_bytes)


    except Exception as e:
        print(f"WebSocket error: {e}")
        await websocket.close()