# Smart Hawker AI Backend

A simple Flask backend scaffold for the Smart Hawker AI project.

## Run locally

1. Create and activate a Python environment
   ```bash
   python -m venv .venv
   source .venv/bin/activate (macOS/Linux)
   .venv\Scripts\Activate.ps1 (Windows) 
   ```
2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```
3. Start the backend
   ```bash
   python main.py
   ```

## Endpoints

- `GET /health` - health check
- `POST /api/recommendation` - get a stall recommendation
- `GET /api/stalls` - get example stall data

## Notes

- The backend uses Flask for API routing and Pydantic for payload validation.
- Replace the stubbed logic in `services.py` with real hawker recommendation and agent orchestration.
- The backend runs on port `8000` by default.
