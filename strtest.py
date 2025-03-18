import torchaudio
from generator import load_csm_1b
import torch
def test_streaming_generator():
    # Initialize the generator
    generator = load_csm_1b(device="cuda")
    
    # Collect audio chunks
    audio_chunks = []
    for chunk in generator.generate(
        text="Hello, this is great.",
        speaker=0,
        context=[],
        max_audio_length_ms=5000,  # 5 seconds for quick test
        temperature=0.9,
        topk=50,
        chunk_size=20
    ):
        audio_chunks.append(chunk.cpu())  # Move to CPU for saving
    
    if audio_chunks:
        # Concatenate all 1D chunks along time dimension
        full_audio = torch.cat(audio_chunks, dim=0)
        # Add channel dimension (shape becomes [1, time])
        full_audio = full_audio.unsqueeze(0)
        # Save to file
        torchaudio.save("test_stream_output.wav", full_audio, generator.sample_rate)
        print(f"Saved audio with {full_audio.shape[1]/generator.sample_rate:.2f} seconds duration")
    else:
        print("No audio generated")

if __name__ == "__main__":
    test_streaming_generator()