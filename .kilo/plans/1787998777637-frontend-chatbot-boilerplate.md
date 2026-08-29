# Frontend Chatbot Boilerplate Plan

## Goal
Create a functional Flask frontend chatbot UI in `frontend/` that can be connected to backend endpoints later. No backend work is needed right now.

## Current State
- `frontend/app.py` returns "Hello World!"
- `frontend/requirements.txt` has only Flask
- No HTML, CSS, or JS exists yet

## Plan

### 1. Update `frontend/requirements.txt`
Keep Flask; no new dependencies needed for a basic chatbot UI.

### 2. Replace `frontend/app.py`
Serve a single-page chatbot with:
- `GET /` → renders the chat UI
- `POST /api/chat` → accepts JSON `{"message": "..."}`, returns a mock response so the frontend is testable before endpoints exist

### 3. Create `frontend/templates/index.html`
A single HTML file containing:
- A clean chatbot layout: header, messages container, input area, send button
- Embedded CSS for a responsive chat bubble design (match project colors if any)
- Embedded JavaScript for:
  - Sending messages to `/api/chat`
  - Displaying user and bot messages
  - Auto-scrolling to the latest message
  - Handling loading/typing states

### 4. Add clear integration comments
In both the JS `fetch` call and the Flask `/api/chat` handler, add `TODO` comments pointing to the real backend endpoint (e.g., `http://localhost:8000/api/recommendation`) so swapping to the real backend is trivial.

### 5. Validation
- Run the frontend locally
- Open `http://localhost:5000`
- Send a message in the chat
- Verify the mock response appears and the UI scrolls correctly
