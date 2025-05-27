from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# Load fine-tuned model and tokenizer
model_path = "./t5_5s_finetuned"  # adjust this path to your saved model
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# FastAPI app
app = FastAPI()

# Define request structure
class QueryRequest(BaseModel):
    question: str

@app.post("/api/ask")
async def ask_question(query: QueryRequest):
    input_text = query.question
    input_ids = tokenizer.encode(input_text, return_tensors="pt", truncation=True, padding=True).to(device)

    # Generate response
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_length=128, num_beams=4, early_stopping=True)
    
    answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return {"answer": answer}
