from fastapi import FastAPI, Depends, HTTPException
from fastapi.exceptions import RequestValidationError
from starlette.responses import JSONResponse
from app.schemas import VoiceDetectionRequest, VoiceDetectionSuccess, VoiceDetectionError
from app.security import get_api_key
from app.inference import classifier
import logging

app = FastAPI(title="AI Voice Detection System")
logger = logging.getLogger("api")

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Global error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"status": "error", "message": "Internal Server Error"}
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=422,
        content={"status": "error", "message": str(exc)}
    )

@app.post("/api/voice-detection", response_model=VoiceDetectionSuccess, responses={400: {"model": VoiceDetectionError}, 401: {"model": VoiceDetectionError}})
async def detect_voice(request: VoiceDetectionRequest, api_key: str = Depends(get_api_key)):
    """
    Detects if the provided MP3 audio is AI-generated or Human.
    """
    try:
        # Perform classification
        classification, confidence, explanation = classifier.predict(request.audioBase64)
        
        return VoiceDetectionSuccess(
            language=request.language.value,
            classification=classification,
            confidenceScore=confidence,
            explanation=explanation
        )

    except ValueError as e:
        logger.warning(f"Bad request: {e}")
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": str(e)}
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": "Analysis failed due to internal error."}
        )

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": classifier.model is not None}
