import torchaudio
from generator import Generator  # Import your Generator class
import time
import torch
from generator import load_csm_1b

def test_streaming(
    text: str = "Hello this is a streaming test",
    speaker: int = 0,
    output_path: str = "stream_test.wav",
    max_audio_length_ms: float = 10_000,
    chunk_size: int = 20
):
    # Initialize generator (use same initialization as your app)
    generator = load_csm_1b(device="cuda")
    
    start_time = time.time()
    
    # Collect all audio chunks
    audio_chunks = []
    audio_gen = generator.generate(
        text=text,
        speaker=speaker,
        context=[],
        max_audio_length_ms=max_audio_length_ms,
        chunk_size=chunk_size
    )
    
    for chunk in audio_gen:
        print(f"Received chunk of {chunk.shape[0]/generator.sample_rate:.2f}s")
        audio_chunks.append(chunk.cpu())
    
    # Combine and save
    if audio_chunks:
        full_audio = torch.cat(audio_chunks, dim=0)
        torchaudio.save(
            output_path,
            full_audio.unsqueeze(0),
            generator.sample_rate,
            encoding="PCM_S",
            bits_per_sample=16
        )
        print(f"Saved {full_audio.shape[0]/generator.sample_rate:.2f}s audio to {output_path}")
    else:
        print("No audio generated")
    
    print(f"Total processing time: {time.time()-start_time:.2f}s")

if __name__ == "__main__":
    test_streaming()
