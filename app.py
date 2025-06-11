from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline
import base64
import io
from PIL import Image
import uvicorn
import json

app = FastAPI()

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model (will download on first run)
# Using a smaller vision-language model that can run on CPU
model = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

@app.get("/")
def root():
    return {"message": "Vision API is running! Send POST requests to /generate"}

@app.post("/generate")
async def generate(request: Request):
    try:
        body = await request.json()
        
        # Get data URL and custom prompt
        data_url = body.get("image", "")
        custom_prompt = body.get("prompt", "Describe ONLY the primary fabric piece using EXACTLY 7-12 words. Include ONLY dominant color(s), texture, pattern, and fit shape.")
        
        # Process data URL
        image_data = data_url.split(",")[1] if "," in data_url else data_url
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Generate description
        result = model(image)[0]["generated_text"]
        
        # Process result to match requirements
        processed_result = result.split(".")[0].strip()
        words = processed_result.split()
        if len(words) > 15:
            processed_result = " ".join(words[:15])
            
        if not processed_result.endswith("."):
            processed_result += "."
            
        return {"description": processed_result}
    
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
