# **API Documentation: Smart Learning Assistant Chatbot**

## **Base URL** *(deployed on render except some latency and cold reboots)*
`[https://<your-api-url>.onrender.com](https://lms-hk98.onrender.com)`

---

## **Endpoints**

### **1. POST /chat**
This endpoint interacts with the chatbot to provide personalized course recommendations and guidance.

---

## **Request**

### **Headers**
| Key           | Value           |
|---------------|-----------------|
| `Content-Type`| `application/json` |

### **Body (JSON)**
| Parameter      | Type   | Description                                                                 |
|----------------|--------|-----------------------------------------------------------------------------|
| `user_profile` | Object | User details, including educational goals and completed courses.            |
| `query`        | String | The user's input query describing what they wish to learn or their goals.   |

#### **Example Payload**
```json
{
  "user_profile": {
    "goal": "Data Science",
    "completed_courses": ["Python for Beginners", "Intro to SQL"]
  },
  "query": "I want to learn about data visualization."
}
```
NOTE : USER_PROFILE has flexible shema anything can be passed
---

## **Response**

### **Success (200 OK)**
Returns a response generated by the chatbot, including a list of relevant course recommendations.

#### **Response Body (JSON)**
| Key               | Type   | Description                                                       |
|-------------------|--------|-------------------------------------------------------------------|
| `response`        | String | Chatbot's personalized guidance based on the user query.         |
| `relevant_courses`| Array  | List of recommended courses, with details.                       |

#### **Example Response**
```json
{
  "response": "Based on your interest in data visualization, I recommend the following courses.",
  "relevant_courses": [
    {
      "course_title": "Data Visualization with Python",
      "rating": "0.9",
      "category": "Data Science",
      "description": "Learn to create stunning visualizations using Python libraries like Matplotlib and Seaborn."
    },
    {
      "course_title": "Introduction to Tableau",
      "rating": "0.85",
      "category": "Business Intelligence",
      "description": "Master Tableau and create impactful dashboards for business insights."
    }
  ]
}
```

### **Error (500 Internal Server Error)**
Returns an error message if something goes wrong while processing the request.

#### **Example Error Response**
```json
{
  "detail": "An error occurred while processing your request."
}
```

---

## **Usage Example**

### **cURL**
```bash
curl -X POST "https://<your-api-url>.onrender.com/chat" \
-H "Content-Type: application/json" \
-d '{
  "user_profile": {
    "goal": "Data Science",
    "completed_courses": ["Python for Beginners", "Intro to SQL"]
  },
  "query": "I want to learn about data visualization."
}'
```

### **JavaScript Fetch**
```javascript
fetch("https://<your-api-url>.onrender.com/chat", {
  method: "POST",
  headers: {
    "Content-Type": "application/json"
  },
  body: JSON.stringify({
    user_profile: {
      goal: "Data Science",
      completed_courses: ["Python for Beginners", "Intro to SQL"]
    },
    query: "I want to learn about data visualization."
  })
})
  .then(response => response.json())
  .then(data => console.log(data))
  .catch(error => console.error("Error:", error));
```

### **Python Requests**
```python
import requests

url = "https://<your-api-url>.onrender.com/chat"
payload = {
    "user_profile": {
        "goal": "Data Science",
        "completed_courses": ["Python for Beginners", "Intro to SQL"]
    },
    "query": "I want to learn about data visualization."
}
headers = {"Content-Type": "application/json"}

response = requests.post(url, json=payload, headers=headers)
print(response.json())
```

---

## **Notes for Developers**
- **Field Validation:** Ensure all required fields are provided in the request payload.
- **Default Values in Response:** Missing course data (e.g., title, rating) will be replaced with defaults such as `"Unnamed Course"` or `"N/A"`.
- **Error Handling:** Always check the HTTP status code and `detail` field in the response to handle errors gracefully.
- **Scalability:** This API is designed to handle multiple queries simultaneously. Optimize your frontend calls to avoid redundant requests.

---

Feel free to contact me for further clarification or enhancements.
