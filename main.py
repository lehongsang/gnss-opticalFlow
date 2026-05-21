import warnings
warnings.filterwarnings("ignore")

from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import uuid
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
MODEL_PATH = "optical_flow_estimation_raft_2023aug_int8bq.onnx"
try:
    processor = OpticalFlowProcessor(MODEL_PATH)
    print(f"Successfully loaded model from {MODEL_PATH}")
except Exception as e:
    print(f"Warning: Failed to load model {MODEL_PATH}. Error: {e}")
    processor = None

TEMP_DIR = "temp_videos"
os.makedirs(TEMP_DIR, exist_ok=True)

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

    # Save uploaded video
    req_id = str(uuid.uuid4())
    input_filename = f"{req_id}_input.mp4"
    output_filename = f"{req_id}_output.mp4"
    
    input_path = os.path.join(TEMP_DIR, input_filename)
    output_path = os.path.join(TEMP_DIR, output_filename)
    
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    vector_direction_sign = 1.0 if is_moving else -1.0
    
    try:
        # Process the video
        processor.process_video(input_path, output_path, mode=mode.upper(), vector_direction_sign=vector_direction_sign)
        
        # Schedule cleanup after sending response
        background_tasks.add_task(cleanup_files, [input_path, output_path])
        
        return FileResponse(
            path=output_path, 
            media_type="video/mp4",
            filename=output_filename
        )
    except Exception as e:
        cleanup_files([input_path, output_path])
        return {"error": str(e)}

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": processor is not None}
