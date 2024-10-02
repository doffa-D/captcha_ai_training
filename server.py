from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import cairosvg
from main import predict_text_from_image, generate_char_preset, idx2char, num_chars

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins. Adjust for production as necessary.
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],  # Allow all headers
)

class SVGData(BaseModel):
    svg: str

# Define the global parameters needed by the model
model_path = 'best_model.pth'  # Ensure you have the correct path to your trained model
allowed_characters = generate_char_preset()
letters = sorted(list(set(allowed_characters)))
vocabulary = ["-"] + letters
num_chars = len(vocabulary)

# Create index mappings
idx2char = {k: v for k, v in enumerate(vocabulary)}
char2idx = {v: k for k, v in idx2char.items()}

@app.post("/convert_svg")
async def upload_svg(data: SVGData):
    try:
        svg_content = data.svg
        # Validate SVG content
        if not svg_content.strip().startswith("<svg"):
            raise HTTPException(status_code=400, detail="Invalid SVG data.")

        # Define the directory to save images
        save_dir = "./"
        os.makedirs(save_dir, exist_ok=True)

        # Convert SVG to PNG directly from string
        png_file_path = os.path.join(save_dir, f"doffa.png")
        cairosvg.svg2png(bytestring=svg_content.encode('utf-8'), write_to=png_file_path)

        # Call the `predict_text_from_image` function with the correct arguments
        predicted_text = predict_text_from_image(png_file_path, model_path, idx2char, num_chars)

        return {
            "predicted_text": f"{predicted_text}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# This is needed for reload to work
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=5000)
