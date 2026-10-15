"""Entry point: ``python main.py`` or ``uvicorn main:app``."""
import uvicorn

from app.main import app

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="127.0.0.1", port=5000, reload=True)
