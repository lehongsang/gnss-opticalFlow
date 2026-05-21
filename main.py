import warnings
warnings.filterwarnings("ignore")

from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import json
import uuid
import tempfile
from inference import OpticalFlowProcessor

app = FastAPI(title="Optical Flow Server")

# Allow CORS for potential web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the model processor
DEFAULT_MODEL = "optical_flow_estimation_raft_2023aug_int8bq.onnx"
DEQUANT_MODEL = "optical_flow_estimation_raft_2023aug_dequant.onnx"
ALT_MODEL = "optical_flow_estimation_raft_2023aug.onnx"
# Prefer dequantized float model, then alternate float model, then default int8 model
if os.path.exists(DEQUANT_MODEL):
    MODEL_PATH = DEQUANT_MODEL
elif os.path.exists(ALT_MODEL):
    MODEL_PATH = ALT_MODEL
else:
    MODEL_PATH = DEFAULT_MODEL
try:
    processor = OpticalFlowProcessor(MODEL_PATH)
    print(f"Successfully loaded model from {MODEL_PATH}")
except Exception as e:
    print(f"Warning: Failed to load model {MODEL_PATH}. Error: {e}")
    processor = None

def cleanup_files(file_paths):
    for path in file_paths:
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception as e:
            print(f"Failed to cleanup {path}: {e}")


@app.post("/process-video")
async def process_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    mode: str = Form("VECTORS"),
    is_moving: bool = Form(False)
):
    """
    Process a video using the RAFT Optical Flow model.
    mode: "VECTORS" or "HEATMAP"
    is_moving: true if the camera is moving forward, false if moving backward/stationary (affects vector direction)
    """
    if not processor:
        return {"error": "Model not loaded on server."}

    # Save uploaded video to system temporary files
    input_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    input_path = input_temp.name
    input_temp.close()
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    output_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    output_path = output_temp.name
    output_temp.close()

    vector_direction_sign = 1.0 if is_moving else -1.0

    try:
        # Process the video (no status file)
        processor.process_video(input_path, output_path, mode=mode.upper(), vector_direction_sign=vector_direction_sign)

        # Schedule cleanup after sending response
        background_tasks.add_task(cleanup_files, [input_path, output_path])

        return FileResponse(
            path=output_path,
            media_type="video/mp4",
            filename=os.path.basename(output_path)
        )
    except Exception as e:
        # cleanup temp files on error
        cleanup_files([input_path, output_path])
        return {"error": str(e)}


@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": processor is not None}
