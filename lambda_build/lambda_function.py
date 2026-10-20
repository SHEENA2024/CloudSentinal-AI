
import json
import logging
from datetime import datetime, timezone
from io import StringIO
from urllib.parse import unquote_plus

import boto3
import pandas as pd

from scripts.inference import predict_anomalies

logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3 = boto3.client("s3")


def lambda_handler(event, context):
    # Support both direct test events and S3 notification events.
    if "Records" in event:
        record = event["Records"][0]
        bucket = record["s3"]["bucket"]["name"]
        key = unquote_plus(record["s3"]["object"]["key"])
    else:
        bucket = event.get("bucket")
        key = event.get("key")

    if not bucket or not key:
        raise ValueError("Provide 'bucket' and 'key', or an S3 event.")

    # Prevent accidentally processing our own output files.
    if key.startswith("predictions/"):
        raise ValueError("Input must not be inside the predictions/ prefix.")

    logger.info("Reading input: s3://%s/%s", bucket, key)

    response = s3.get_object(Bucket=bucket, Key=key)
    metrics = pd.read_csv(response["Body"])

    # Run the existing, tested inference pipeline.
    predictions = predict_anomalies(metrics)

    # Serialize every prediction row, including all available input
    # columns, prediction labels, and anomaly scores.
    csv_buffer = StringIO()
    predictions.to_csv(csv_buffer, index=False)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_key = f"predictions/cloudsentinel_predictions_{timestamp}_{context.aws_request_id}.csv"

    s3.put_object(
        Bucket=bucket,
        Key=output_key,
        Body=csv_buffer.getvalue().encode("utf-8"),
        ContentType="text/csv",
        ServerSideEncryption="AES256",
    )

    anomaly_count = int(predictions["is_anomaly_prediction"].sum())

    preview = []
    for row in predictions[
        ["resource_id", "prediction_label", "anomaly_score"]
    ].head(10).to_dict("records"):
        preview.append({
            "resource_id": str(row["resource_id"]),
            "prediction_label": str(row["prediction_label"]),
            "anomaly_score": float(row["anomaly_score"]),
        })

    logger.info(
        "Processed %d rows; predicted %d anomalies; output s3://%s/%s",
        len(predictions),
        anomaly_count,
        bucket,
        output_key,
    )

    return {
        "status": "success",
        "input_bucket": bucket,
        "input_key": key,
        "output_bucket": bucket,
        "output_key": output_key,
        "records_processed": int(len(predictions)),
        "predicted_anomalies": anomaly_count,
        "predictions_preview": preview,
    }
