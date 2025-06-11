from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import base64
import io
import uvicorn
import numpy as np

app = FastAPI()

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load a lightweight model for image feature extraction
# ResNet18 is much smaller than full vision-language models
model = models.resnet18(pretrained=True)
model.eval()  # Set to evaluation mode

# Remove the final classification layer
feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])

# Preprocessing transformation
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Color and texture mapping based on features
color_map = {
    0: "red", 1: "orange", 2: "yellow", 3: "green", 
    4: "blue", 5: "purple", 6: "pink", 7: "white", 
    8: "black", 9: "gray", 10: "brown", 11: "beige"
}

texture_map = {
    0: "smooth", 1: "soft", 2: "rough", 3: "crisp", 
    4: "silky", 5: "shiny", 6: "matte", 7: "textured"
}

pattern_map = {
    0: "solid", 1: "striped", 2: "dotted", 3: "floral", 
    4: "checkered", 5: "patterned"
}

shape_map = {
    0: "fitted", 1: "loose", 2: "flowing", 3: "structured", 
    4: "tailored", 5: "draped"
}

@app.get("/")
def root():
    return {"message": "Lightweight Vision API is running! Send POST requests to /generate"}

@app.post("/generate")
async def generate(request: Request):
    try:
        body = await request.json()
        
        # Get data URL
        data_url = body.get("image", "")
        
        # Process data URL
        image_data = data_url.split(",")[1] if "," in data_url else data_url
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Preprocess and extract features
        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0)  # Add batch dimension
        
        with torch.no_grad():
            features = feature_extractor(input_batch)
            features = features.squeeze().numpy()
        
        # Simple algorithm to extract dominant colors and textures from features
        flattened = features.flatten()
        normalized = (flattened - flattened.min()) / (flattened.max() - flattened.min())
        
        # Get dominant color indices
        color_indices = np.argsort(normalized[-12:])[-2:]  # Get top 2 color indices
        colors = [color_map[idx % 12] for idx in color_indices]
        
        # Get texture
        texture_index = np.argmax(normalized[:8]) % 8
        texture = texture_map[texture_index]
        
        # Get pattern
        pattern_index = np.argmax(normalized[8:14]) % 6
        pattern = pattern_map[pattern_index]
        
        # Get shape
        shape_index = np.argmax(normalized[14:20]) % 6
        shape = shape_map[shape_index]
        
        # Format the description
        if pattern == "solid":
            description = f"{colors[0]} {texture} {shape}."
        else:
            description = f"{colors[0]} and {colors[1]} {pattern} {texture} {shape}."
            
        return {"description": description}
    
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
