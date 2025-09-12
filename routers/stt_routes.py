import os
from fastapi import APIRouter, UploadFile, File, Form, status, HTTPException
from services.whisper_runtime import get_stt_model
from utils.error_codes import ErrorCodes
from utils.response import success_response, error_response
from services.whisper_services import transcribe_upload

router = APIRouter(prefix="/stt", tags=["stt"])

@router.post("/transcribe")
async def transcribe(
    audio: UploadFile = File(..., description="Audio file (wav/m4a/mp3/etc.)"),
    language: str = Form(default="en", description="ISO code or auto-detect if omitted")
):
    """
    Success: JSON { success, message, data: { text, language, language_probability, segments[] } }
    Error:   JSON error via your standardized error wrapper.
    """
    try:
        result = transcribe_upload(audio, language=language)
        return success_response(
            data=result,
            message="Transcription successful",
            status_code=status.HTTP_200_OK
        )
    except HTTPException as he:
        code = ErrorCodes.BAD_REQUEST if he.status_code in (status.HTTP_400_BAD_REQUEST,
                                                           status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                                                           status.HTTP_422_UNPROCESSABLE_ENTITY) \
               else ErrorCodes.SERVER_ERROR
        return error_response(
            message=str(he.detail),
            error_code=code,
            status_code=he.status_code
        )
    except Exception as e:
        return error_response(
            message="Failed to transcribe audio.",
            error_code=ErrorCodes.SERVER_ERROR,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@router.get("/health")
async def stt_health():
    """
    Health check for faster-whisper.
    Verifies the model singleton loads and returns config info.
    """
    try:
        model = get_stt_model()
        meta = {
            "loaded": True,
            "engine": "faster-whisper",
            "model": os.getenv("WHISPER_MODEL", "small.en"),
            "device": os.getenv("WHISPER_DEVICE", "cpu"),
            "compute_type": os.getenv(
                "WHISPER_COMPUTE_TYPE",
                "float16" if os.getenv("WHISPER_DEVICE", "cuda") == "cuda" else "int8"
            )
        }
        return success_response(data=meta, message="STT healthy", status_code=status.HTTP_200_OK)
    except Exception as e:
        return error_response(
            message=f"STT not healthy: {e}",
            error_code=ErrorCodes.SERVER_ERROR,
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE
        )