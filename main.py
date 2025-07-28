from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import boto3
from botocore.exceptions import BotoCoreError, ClientError
import os

# Create FastAPI app
app = FastAPI(title="Face Compare Service")

# Initialize boto3 client (region from ENV)
rekognition = boto3.client(
    "rekognition",
    region_name=os.getenv("AWS_REGION", "us-east-1")
)

class CompareRequest(BaseModel):
    source_bucket: str
    source_key: str
    target_bucket: str
    target_key: str
    similarity_threshold: float = 90.0

@app.post("/compare_faces")
def compare_faces(req: CompareRequest):
    try:
        response = rekognition.compare_faces(
            SourceImage={'S3Object': {'Bucket': req.source_bucket, 'Name': req.source_key}},
            TargetImage={'S3Object': {'Bucket': req.target_bucket, 'Name': req.target_key}},
            SimilarityThreshold=req.similarity_threshold
        )
        # Return only key information
        return {
            "matches": [
                {
                    "similarity": m["Similarity"],
                    "bounding_box": m["Face"]["BoundingBox"]
                } for m in response.get("FaceMatches", [])
            ],
            "unmatched_faces": len(response.get("UnmatchedFaces", []))
        }

    except (BotoCoreError, ClientError) as e:
        raise HTTPException(status_code=500, detail=str(e))
