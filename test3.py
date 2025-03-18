import uuid
from fastapi import FastAPI, WebSocket
import torch
from generator import load_csm_1b, Segment
import torchaudio
import numpy as np
import asyncio
from concurrent.futures import ThreadPoolExecutor

app = FastAPI()
executor = ThreadPoolExecutor()
# Load generator (consider caching in real applications)
generator = load_csm_1b(device="cuda")

def load_audio(audio_path):
    audio_tensor, sample_rate = torchaudio.load(audio_path)
    audio_tensor = torchaudio.functional.resample(
        audio_tensor.squeeze(0), orig_freq=sample_rate, new_freq=generator.sample_rate
    )
    return audio_tensor

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    speakers = [0]
    transcripts = ["Excuse me, I specifically ordered for a vegetarian meal. Is there an issue with my meal request?",]
    audio_paths = ["generated_utterances/init_male_smooth.wav",]
    try:
        while True:
            params = await websocket.receive_json()
            text_to_store_as_context = params.get("speaker 1", "Yes there seems to be an issue with your meal request. May I know what did you originally ordered?")
            audio_to_store_as_context = params.get("speaker 1 audio path", "utterances/uid.wav")
            text_to_generate = params.get("speaker 0", "I ordered a singaporean vegetarian briyani.")
            

            # Generate unique ID and create paths
            unique_id = uuid.uuid4()
            output_path = f"generated_utterances/{unique_id}.wav"

            # Update the speaker list, transcript list and audio_paths list.
            # speakers.append(1)
            # transcripts.append(text_to_store_as_context)
            # audio_paths.append(audio_to_store_as_context)

            # Generate Segment for context
            segments = [
                Segment(text=transcript, speaker=speaker, audio=load_audio(audio_path))
                for transcript, speaker, audio_path in zip(transcripts, speakers, audio_paths)
            ]
            

            audio_chunks = []

            loop = asyncio.get_event_loop()
            
            # Create generator instance
            audio_gen = generator.generate(
                text=text_to_generate,
                speaker=0,
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

                # Update the speaker list, transcript list and audio_paths list.
                speakers.append(0)
                transcripts.append(text_to_generate)
                audio_paths.append(output_path)

    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await websocket.close()
