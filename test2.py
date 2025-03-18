import uuid
from fastapi import FastAPI, WebSocket
import torch
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
            params = await websocket.receive_json()
            text = params.get("text", "Hello from Sesame.")
            speaker = params.get("speaker", 0)

            # Generate unique ID and create paths
            unique_id = uuid.uuid4()
            output_path = f"utterances/{unique_id}.wav"
            audio_chunks = []

            loop = asyncio.get_event_loop()
            
            # Create generator instance
            audio_gen = generator.generate(
                text=text,
                speaker=speaker,
                context=[],
                max_audio_length_ms=10_000,
                temperature=0.9,
                topk=50,
                chunk_size=20  # Match parameter name
            )

            # Create bridge between sync generator and async WebSocket
            def sync_generator():
                for audio_chunk in audio_gen:
                    audio_np = audio_chunk.numpy().astype(np.float32)
                    audio_chunks.append(audio_chunk.cpu())
                    yield audio_np.tobytes()

            # Process chunks using thread pool
            gen = sync_generator()
            for chunk_bytes in gen:
                await websocket.send_bytes(chunk_bytes)

            # Save full audio after streaming completes
            if audio_chunks:
                full_audio = torch.cat(audio_chunks, dim=0)
                torchaudio.save(
                    output_path,
                    full_audio.unsqueeze(0),
                    generator.sample_rate,
                    encoding="PCM_S",
                    bits_per_sample=16
                )
                print(f"Saved audio to {output_path}")

    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await websocket.close()
