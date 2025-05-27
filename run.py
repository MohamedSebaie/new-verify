import uvicorn # type: ignore
import logging

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=4007,
        reload=True,
        log_level="info"
    )