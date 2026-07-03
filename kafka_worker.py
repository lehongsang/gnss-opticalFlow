import os
# Fix for boto3/botocore chunked upload corruption on S3-compatible storage through Nginx
os.environ["AWS_REQUEST_CHECKSUM_CALCULATION"] = "WHEN_REQUIRED"
os.environ["AWS_RESPONSE_CHECKSUM_CALCULATION"] = "WHEN_REQUIRED"

import json
import logging
import tempfile
from dotenv import load_dotenv
import boto3
from botocore.config import Config
from kafka import KafkaConsumer, KafkaProducer

# Load environment variables
load_dotenv()

# Setup Logging
LOG_LEVEL = os.getenv("OPTICAL_FLOW_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("optical_flow.worker")

# Load AI model processor
from inference import OpticalFlowProcessor, ProcessingCancelled

FP16_MODEL = "optical_flow_estimation_raft_2023aug_fp16.onnx"
DEFAULT_MODEL = "optical_flow_estimation_raft_2023aug_int8bq.onnx"
DEQUANT_MODEL = "optical_flow_estimation_raft_2023aug_dequant.onnx"
ALT_MODEL = "optical_flow_estimation_raft_2023aug.onnx"

if os.path.exists(FP16_MODEL):
    MODEL_PATH = FP16_MODEL
elif os.path.exists(DEQUANT_MODEL):
    MODEL_PATH = DEQUANT_MODEL
elif os.path.exists(ALT_MODEL):
    MODEL_PATH = ALT_MODEL
else:
    MODEL_PATH = DEFAULT_MODEL

try:
    logger.info(f"Loading model from path={MODEL_PATH}")
    processor = OpticalFlowProcessor(MODEL_PATH)
    logger.info("Successfully loaded model")
except Exception as e:
    logger.exception(f"Failed to load model: {e}")
    processor = None

# S3 Client Setup
S3_ENDPOINT = os.getenv("S3_ENDPOINT", "http://localhost:8333")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY", "change-me-s3-access-key")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY", "change-me-s3-secret-key")
S3_BUCKET = os.getenv("S3_BUCKET", "medias")
S3_REGION = os.getenv("S3_REGION", "us-east-1")
S3_USE_SSL = os.getenv("S3_USE_SSL", "False").lower() == "true"
S3_FORCE_PATH_STYLE = os.getenv("S3_FORCE_PATH_STYLE", "False").lower() == "true"

s3_client = boto3.client(
    "s3",
    endpoint_url=S3_ENDPOINT,
    aws_access_key_id=S3_ACCESS_KEY,
    aws_secret_access_key=S3_SECRET_KEY,
    region_name=S3_REGION,
    use_ssl=S3_USE_SSL,
    config=Config(
        signature_version="s3v4",
        s3={
            "addressing_style": "path" if S3_FORCE_PATH_STYLE else "auto",
            "payload_signing_enabled": False
        }
    ),
)

# Kafka Configuration
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_CLIENT_ID = os.getenv("KAFKA_CLIENT_ID", "gnss-ai-worker")
KAFKA_JOBS_TOPIC = os.getenv("KAFKA_JOBS_TOPIC", "gnss.media.process.jobs")
KAFKA_RESULTS_TOPIC = os.getenv("KAFKA_RESULTS_TOPIC", "gnss.media.process.results")
KAFKA_ALERTS_TOPIC = os.getenv("KAFKA_ALERTS_TOPIC", "gnss.alerts")
KAFKA_CONSUMER_GROUP = os.getenv("KAFKA_CONSUMER_GROUP", "gnss.media.process.jobs.group")

# Setup Kafka Producer
producer = KafkaProducer(
    bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
    client_id=KAFKA_CLIENT_ID,
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

def send_result(job_id, status, output_s3_key=None, error_msg=None):
    payload = {
        "jobId": job_id,
        "status": status,
    }
    if output_s3_key:
        payload["outputS3Key"] = output_s3_key
    if error_msg:
        payload["error"] = error_msg

    try:
        # Send message to results topic
        future = producer.send(
            KAFKA_RESULTS_TOPIC,
            key=job_id.encode('utf-8') if job_id else None,
            value=payload
        )
        producer.flush()
        future.get(timeout=10) # Block until message is delivered
        logger.info(f"Published job result to Kafka: {payload}")
    except Exception as e:
        logger.error(f"Failed to publish job result to Kafka: {e}")

def send_motion_alerts(job_id, device_id, alerts):
    """Send detected motion alerts to the backend via Kafka.

    Each alert is published as a separate message to the alerts topic.
    The backend's alerts.consumer.ts will pick these up, auto-lookup
    the latest GPS coordinates from telemetry, save to DB, and
    broadcast via WebSocket.
    """
    for alert in alerts:
        payload = {
            "payload": {
                "deviceId": device_id,
                "type": alert.get("type", "sudden_motion"),
                "severity": alert.get("severity", "HIGH"),
                "message": alert.get("message", "Motion anomaly detected by AI"),
                "location": {"lat": 0, "lng": 0},  # Backend will auto-fill from latest telemetry
                "snapshotId": None,
            }
        }
        try:
            future = producer.send(
                KAFKA_ALERTS_TOPIC,
                key=device_id.encode("utf-8") if device_id else None,
                value=payload,
            )
            producer.flush()
            future.get(timeout=10)
            logger.info(
                "Published motion alert to Kafka topic=%s device=%s type=%s",
                KAFKA_ALERTS_TOPIC,
                device_id,
                alert.get("type"),
            )
        except Exception as e:
            logger.error(
                "Failed to publish motion alert to Kafka: %s alert=%s", e, alert
            )

def process_job(job_data):
    job_id = job_data.get("jobId")
    input_s3_key = job_data.get("inputS3Key")
    mode = job_data.get("mode", "VECTORS").upper()
    is_moving = job_data.get("isMoving", True)
    
    if not job_id or not input_s3_key:
        logger.error(f"Invalid job payload missing jobId or inputS3Key: {job_data}")
        return

    if not processor:
        logger.error("Optical Flow processor is not loaded.")
        send_result(job_id, "failed", error_msg="Model not loaded on AI worker.")
        return

    logger.info(f"Starting job={job_id} for key={input_s3_key} mode={mode} is_moving={is_moving}")
    
    # Create temp files for input and output
    input_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    input_path = input_temp.name
    input_temp.close()

    output_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    output_path = output_temp.name
    output_temp.close()

    try:
        # Step 1: Download from S3
        logger.info(f"Downloading s3://{S3_BUCKET}/{input_s3_key} to {input_path}")
        s3_client.download_file(S3_BUCKET, input_s3_key, input_path)
        
        # Step 2: Process video
        vector_direction_sign = 1.0 if is_moving else -1.0
        logger.info("Running AI optical flow processor...")
        detected_alerts = processor.process_video(
            input_path,
            output_path,
            mode=mode,
            vector_direction_sign=vector_direction_sign,
            req_id=job_id,
            is_moving=is_moving,
        )

        # Step 3: Upload result to S3
        device_id = job_data.get("deviceId", "unknown")
        output_s3_key = f"media-logs/{device_id}/processed_{job_id}.mp4"
        logger.info(f"Uploading result {output_path} to s3://{S3_BUCKET}/{output_s3_key}")
        
        s3_client.upload_file(
            output_path,
            S3_BUCKET,
            output_s3_key,
            ExtraArgs={'ContentType': 'video/mp4'}
        )

        # Step 4: Report Success
        send_result(job_id, "completed", output_s3_key=output_s3_key)
        logger.info(f"Successfully finished job={job_id}")

        # Step 5: Send detected motion alerts to backend
        if detected_alerts:
            logger.info(
                f"Sending {len(detected_alerts)} motion alert(s) for job={job_id} device={device_id}"
            )
            send_motion_alerts(job_id, device_id, detected_alerts)
        else:
            logger.info(f"No motion anomalies detected for job={job_id}")

    except Exception as e:
        logger.exception(f"Error processing job={job_id}")
        send_result(job_id, "failed", error_msg=str(e))
    finally:
        # Cleanup temp files
        for path in (input_path, output_path):
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception as ex:
                logger.warning(f"Failed to delete temp file {path}: {ex}")

def main():
    consumer = KafkaConsumer(
        KAFKA_JOBS_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        group_id=KAFKA_CONSUMER_GROUP,
        client_id=KAFKA_CLIENT_ID,
        auto_offset_reset='earliest',
        enable_auto_commit=True,
        value_deserializer=lambda m: json.loads(m.decode('utf-8')),
        max_poll_interval_ms=3600000 # 1 hour (3,600,000 ms)
    )
    
    logger.info(f"Kafka worker started, listening on topic={KAFKA_JOBS_TOPIC} server={KAFKA_BOOTSTRAP_SERVERS}")

    try:
        for message in consumer:
            try:
                job_data = message.value
                process_job(job_data)
            except Exception as e:
                logger.error(f"Failed to parse or run job message: {e}")

    except KeyboardInterrupt:
        logger.info("Kafka worker stopped by keyboard interrupt.")
    finally:
        consumer.close()

if __name__ == "__main__":
    main()
