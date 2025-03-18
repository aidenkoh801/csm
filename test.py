import os
from typing import List
import torchaudio
import torch
from fastapi import FastAPI, WebSocket
from generator import load_csm_1b, Segment

app = FastAPI()

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(device)

generator = load_csm_1b(device=device)

def load_audio(audio_path):
    audio_tensor, sample_rate = torchaudio.load(audio_path)
    audio_tensor = torchaudio.functional.resample(
        audio_tensor.squeeze(0), 
        orig_freq=sample_rate, 
        new_freq=generator.sample_rate
    )
    return audio_tensor

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    # Initialize context for each WebSocket connection
    speakers: List[int] = [1]
    transcripts: List[str] = [
    "Excuse me, I specifically ordered for a vegetarian meal. Is there any issue with my meal request?",
    ]
    audio_paths: List[str] = [
    "ini.wav",
    ]

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            print(generator.sample_rate)
            speaker_0_text = data.get('speaker_0_text')
            # The JSON only contains the unique audio filename
            speaker_0_audio_filename = data.get('speaker_0_audio_path')
            # Prepend the "utterances" folder path
            speaker_0_audio_path = os.path.join("utterances", speaker_0_audio_filename)

            speaker_1_text = data.get('speaker_1_text')

            # Append speaker 0's data to context
            speakers.append(0)
            transcripts.append(speaker_0_text)
            audio_paths.append(speaker_0_audio_path)

            # Build segments
            segments = []
            for i in range(len(audio_paths)):
                audio = load_audio(audio_paths[i])
                segments.append(
                    Segment(
                        text=transcripts[i],
                        speaker=speakers[i],
                        audio=audio
                    )
                )
            print("generating..")
            # Generate speaker 1's audio
            new_audio = generator.generate(
                text=speaker_1_text,
                speaker=1,
                context=segments,
                max_audio_length_ms=10_000,
            )
            print("generated.")

            # Save the generated audio
            #new_audio_path = f"generated_{len(audio_paths)}.wav"
            new_audio_path = os.path.join("generated", f"generated_{len(audio_paths)}.wav")
            torchaudio.save(new_audio_path, new_audio.unsqueeze(0).cpu(), generator.sample_rate)

            # Append speaker 1's data to context
            speakers.append(1)
            transcripts.append(speaker_1_text)
            audio_paths.append(new_audio_path)

            # Send back the path to the generated audio
            await websocket.send_json({"speaker_1_audio_path": new_audio_path})

    except Exception as e:
        print(f"WebSocket error: {e}")
        await websocket.close()
