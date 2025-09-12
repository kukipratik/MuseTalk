import os
import tempfile
from typing import Optional, Dict, Any
from fastapi import UploadFile, HTTPException, status

from services.whisper_runtime import get_stt_model

def transcribe_upload(file: UploadFile, language: Optional[str] = None) -> Dict[str, Any]:
    if file is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No file provided.")
    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Filename missing.")
    if not (file.content_type and file.content_type.startswith("audio/")):
        allowed = {"audio/", "application/octet-stream"}
        if not any(file.content_type and file.content_type.startswith(p) for p in allowed):
            raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                                detail=f"Unsupported content-type: {file.content_type}")

    try:
        suffix = os.path.splitext(file.filename)[-1] or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file.file.read())
            tmp_path = tmp.name
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to buffer upload: {e}")

    try:
        model = get_stt_model()
        segments, info = model.transcribe(
            tmp_path,
            beam_size=1,
            vad_filter=True,
            language=language,
        )
        full_text, seg_list = [], []
        for seg in segments:
            full_text.append(seg.text)
            seg_list.append({"id": seg.id, "start": seg.start, "end": seg.end, "text": seg.text})
        return {
            "text": "".join(full_text).strip(),
            "language": info.language,
            "language_probability": info.language_probability,
            "segments": seg_list
        }
        
        # segments, info = model.transcribe(
        #             tmp_path,
        #             language="en",
        #             beam_size=1,
        #             vad_filter=True,
        #             without_timestamps=True
        #         )
        # text = "".join([seg.text for seg in segments]).strip()
        # return {
        #     "text": text.strip(),
        #     "language": info.language
        # }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")
    finally:
        try: os.remove(tmp_path)
        except: pass
