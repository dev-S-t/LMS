from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import pandas as pd


from dotenv import load_dotenv
import os

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

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load pre-saved data and embeddings
logging.info("Loading course metadata and FAISS index...")
with open("course_metadata.pkl", "rb") as metadata_file:
    course_metadata = pickle.load(metadata_file)

index = faiss.read_index("course_index.faiss")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Configure Gemini API
logging.info("Configuring Gemini API...")
genai.configure(api_key=gemini_key)  # Replace with your actual API key

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
        "You are a **smart learning assistant chatbot** embedded in a **Learning Management System (LMS)**. Your role is to provide personalized, context-aware guidance "
        "to users based on their educational goals and current progress.\n\n"
        "### **Capabilities:**\n"
        "1. **Personalized Course Guidance**: Recommend courses, suggest learning paths, and tailor responses based on user profiles.\n"
        "2. **Real-Time Support**: Resolve user doubts and engage in meaningful, conversational interactions.\n"
        "3. **Interactive Learning**: Generate quizzes and provide feedback to enhance learning.\n"
        "4. **Efficient Information Processing**: Use pre-filtered course data to keep responses relevant and concise.\n\n"
        "**Responsibilities:** Maintain context, adapt dynamically to user inputs, and make learning engaging while keeping responses concise and informative. Also, max rating is 1 which represents 100%, and 0.7 and above are high ratings."
    ),
)

# Pydantic model for input payload
class ChatRequest(BaseModel):
    user_profile: dict
    query: str

# Helper functions

def find_relevant_courses(query, top_n=5):
    """Retrieve the most relevant courses for a given user query, handling missing or NaN values."""
    query_embedding = embedder.encode([query])[0]
    distances, indices = index.search(np.array([query_embedding]), top_n)
    
    relevant_courses = []
    for idx in indices[0]:
        course = course_metadata[idx]
        # Replace NaN or missing values with default values
        sanitized_course = {
            "course_title": course.get("course_title", "Unnamed Course"),
            "rating": course.get("rating", "N/A") if not pd.isna(course.get("rating")) else "N/A",
            "category": course.get("category", "Uncategorized") if not pd.isna(course.get("category")) else "Uncategorized",
            "description": course.get("description", "No description available") if not pd.isna(course.get("description")) else "No description available",
        }
        relevant_courses.append(sanitized_course)
    
    return relevant_courses



def generate_response(user_query, relevant_courses, user_profile):
    """Generate response using the LLM."""
    course_summaries = "\n".join(
        [f"{course.get('course_title', 'Unnamed Course')} "
         f"(Rating: {course.get('rating', 'N/A')}/5, Category: {course.get('category', 'N/A')})"
         for course in relevant_courses]
    )

    context_info = f"User Profile: {user_profile}" if user_profile else "No additional user profile provided."

    prompt = (
        f"User Query: {user_query}\n\n"
        f"{context_info}\n\n"
        f"Relevant Courses:\n{course_summaries}\n\n"
        "Based on the user’s input and available courses, provide personalized guidance on the most suitable courses to take. "
        "Explain why these courses are appropriate and suggest the next steps in the learning journey."
    )

    logging.debug(f"Generated Prompt: {prompt}")

    chat_session = model.start_chat(
        history=[
            {"role": "user", "parts": [user_query]},
            {"role": "model", "parts": ["Hello! How can I assist you today?"]},
        ]
    )
    response = chat_session.send_message(prompt)
    return response.text


# API endpoint
@app.post("/chat")
def chat(request: ChatRequest):
    logging.info("Received chat request")
    logging.info(f"User Profile: {request.user_profile}")
    logging.info(f"Query: {request.query}")

    try:
        # Find relevant courses
        logging.info("Finding relevant courses...")
        relevant_courses = find_relevant_courses(request.query, top_n=5)

        # Generate response
        logging.info("Generating response...")
        response = generate_response(request.query, relevant_courses, request.user_profile)

        logging.info("Response generated successfully.")
        return {"response": response, "relevant_courses": relevant_courses}
    except Exception as e:
        logging.error(f"Error during chat processing: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while processing your request.")

