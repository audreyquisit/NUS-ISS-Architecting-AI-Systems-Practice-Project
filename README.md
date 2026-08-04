# NUS-ISS-Architecting-AI-Systems-Practice-Project
 
# Smart Hawker AI

**Practice Module for Certificate in Architecting AI Systems**

**Team 08**

- Elisha Encinas Zacarias
- Lee Xue Er
- Quisit Jocel Audrey Lugtu
- Jasper Teo Teck Siong

---

# Overview

Smart Hawker AI is a multi-agent AI system that helps users discover suitable hawker stalls in Singapore based on their location, dietary preferences, budget, available time, weather conditions, and estimated queue lengths.

Instead of relying on multiple applications such as Google Maps, GrabFood, or Burpple, Smart Hawker AI provides a single, explainable recommendation by coordinating specialised AI agents and deterministic services.

This project is developed as part of the **Practice Module for the Certificate in Architecting AI Systems** and demonstrates the application of agentic AI architecture, orchestration, responsible AI, DevOps, MLOps, and LLMOps principles.

---

# Tech Stack

- Frontend: Flask (`frontend/app.py`) with a simple Python web app and routing.
- Backend: Flask API (`backend/main.py`) with Pydantic validation for request/response models.
- Containerization: Dockerfiles for frontend and backend services.

## Run the project

### Frontend

```bash
cd frontend
python app.py
```

The frontend currently exposes a simple Flask app on port `5000`.

### Backend

```bash
cd backend
python main.py
```

The backend exposes the API on port `8000` and includes `GET /health`, `GET /api/stalls`, and `POST /api/recommendation`.

---

# Problem Statement

Choosing where to eat at a hawker centre often requires users to consult several disconnected sources for:

- Walking distance
- Stall opening hours
- Menu availability
- Dietary suitability
- Price range
- Queue length
- Weather conditions

This fragmented experience increases decision time and makes personalised recommendations difficult.

Smart Hawker AI addresses this challenge by combining these factors into a single recommendation workflow.

---

# Objectives

The project aims to:

- Provide personalised hawker stall recommendations.
- Demonstrate a multi-agent AI architecture.
- Produce explainable recommendations.
- Prevent hallucinated recommendations through verification.
- Apply responsible AI and secure system design.
- Showcase modern DevOps, MLOps, and LLMOps practices.
