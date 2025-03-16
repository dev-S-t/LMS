from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
import google.generativeai as genai
from dotenv import load_dotenv
import os
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables from the .env file
load_dotenv()

# Retrieve the secret
gemini_key = os.getenv("GEMINI_KEY")
if gemini_key:
    print("Gemini Key retrieved successfully!")
else:
    print("Gemini Key not found.")

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Configure Gemini API
logging.info("Configuring Gemini API...")
genai.configure(api_key=gemini_key)

generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 1024,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
    system_instruction=(
        "**Assistant Role Description**:\n\n"
        "You are a helpful chatbot that assists users with Rental Agreement Complexity. "
        "Help them with their queries and provide search results for the most updated information related to their region. "
        "Always ask them about their region if it is not specified."
    ),
)

# Pydantic model for input payload
class ChatRequest(BaseModel):
    query: str
    region: str = None  # Optional field to capture user region

# API endpoint for chat
@app.post("/chat")
def chat(request: ChatRequest):
    logging.info("Received chat request")
    logging.info(f"Query: {request.query}")
    logging.info(f"Region: {request.region}")

    try:
        # If region is not provided, ask for it in the prompt
        if not request.region:
            prompt = request.query + "\n\nCould you please specify your region?"
        else:
            prompt = request.query + f"\n\nRegion: {request.region}"
        
        # Start a new chat session with a greeting
        chat_session = model.start_chat(
            history=[
                {"role": "user", "parts": [prompt]},
                {"role": "model", "parts": ["Hello! How can I assist you with rental agreement complexity today?"]},
            ]
        )
        response = chat_session.send_message(prompt)
        return {"response": response.text}
    except Exception as e:
        logging.error(f"Error during chat processing: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while processing your request.")
