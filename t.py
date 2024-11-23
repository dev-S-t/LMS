import requests
import json

# Define the API endpoint
API_URL = "http://127.0.0.1:8000/chat"

# Define the test payload
test_payload = {
    "user_profile": {
        "goal": "Data Science",
        "completed_courses": ["Python for Beginners"]
    },
    "query": "I want to learn data visualization."
}

def test_chatbot_api():
    """Send a test request to the chatbot API and display results."""
    print("Sending test payload to the API...")
    print(f"Payload: {json.dumps(test_payload, indent=2)}\n")
    
    try:
        # Send POST request
        response = requests.post(API_URL, json=test_payload)

        # Check if the request was successful
        if response.status_code == 200:
            print("API Response:")
            print(json.dumps(response.json(), indent=2))
        else:
            print(f"Error: Received status code {response.status_code}")
            print(f"Details: {response.text}")

    except Exception as e:
        print(f"An error occurred while testing the API: {e}")

if __name__ == "__main__":
    test_chatbot_api()
