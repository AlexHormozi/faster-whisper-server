import asyncio
from fastapi import FastAPI, WebSocket
from faster_whisper import WhisperModel
import numpy as np

app = FastAPI()

# Load Faster Whisper model (use "cuda" if you have a GPU)
model = WhisperModel("base", device="cpu")

@app.websocket("/transcribe")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected.")
    buffer = b""

    try:
        while True:
            # Receive audio data from the client
            audio_chunk = await websocket.receive_bytes()
            buffer += audio_chunk

            # Process audio when buffer is large enough
            if len(buffer) > 16000 * 2:  # Enough data for 1 second of audio at 16kHz
                audio_data = np.frombuffer(buffer, dtype=np.float32)
                segments, _ = model.transcribe(audio_data)

                # Send transcription back to client
                transcription = " ".join([segment.text for segment in segments])
                await websocket.send_text(transcription)

                # Clear buffer after processing
                buffer = b""
    except Exception as e:
        print(f"Error: {e}")
        await websocket.close()
        print("Client disconnected.")
