# api.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from diffusers.utils import export_to_gif
from model_loader import load_model
import base64
import io

app = FastAPI()

# Load the model
pipe = load_model()

class AnimationRequest(BaseModel):
    prompt: str
    negative_prompt: str = "bad quality, worse quality, low resolution"
    num_frames: int = 16
    guidance_scale: float = 2.0
    num_inference_steps: int = 6
    seed: int = 0

@app.post("/animatelcm")
async def generate_animation(request: AnimationRequest):
    try:
        output = pipe(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            num_frames=request.num_frames,
            guidance_scale=request.guidance_scale,
            num_inference_steps=request.num_inference_steps,
            generator=torch.Generator("cpu").manual_seed(request.seed),
        )
        
        frames = output.frames[0]
        
        # Save the animation to a bytes buffer
        buffer = io.BytesIO()
        export_to_gif(frames, buffer)
        buffer.seek(0)
        
        # Encode the GIF as base64
        gif_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        return {"animation": gif_base64}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7772)
