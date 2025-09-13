from fastapi import status
from fastapi.responses import JSONResponse

def success_response(data=None, message="Success", status_code=status.HTTP_200_OK):
    return JSONResponse(
        status_code=status_code,
        content={
            "success": True,
            "message": message,
            "data": data
        }
    )

def error_response(message, error_code, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, errors=None):
    content = {
        "success": False,
        "message": message,
        "error_code": error_code
    }
    if errors:
        content["errors"] = errors
    return JSONResponse(
        status_code=status_code,
        content=content
    )

def ws_success_response(data=None, message="Success"):
    return {
        "success": True,
        "message": message,
        "data": data
    }

def ws_error_response(message, error_code):
    return {
        "success": False,
        "message": message,
        "error_code": error_code
    }