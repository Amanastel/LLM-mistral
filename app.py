from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Initialize FastAPI
app = FastAPI()

# Define request and response models
class TextRequest(BaseModel):
    prompt: str

class TextResponse(BaseModel):
    generated_text: str  # Renamed for clarity

# Load the model and tokenizer (Publicly available Mistral model)
model_name = "mistralai/Mistral-7B-v0.1"  # Mistral 7B model (base model, publicly available)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Define the endpoint
@app.post("/generate", response_model=TextResponse)
def generate_text(request: TextRequest):  # Renamed for clarity
    try:
        inputs = tokenizer(request.prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_length=100)  # Use max_length instead of max_new_tokens
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return TextResponse(generated_text=generated_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the FastAPI app with uvicorn (if running this script directly)
if _name_ == "_main_":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)