from fastapi import FastAPI, File, UploadFile, HTTPException
import boto3
from botocore.exceptions import BotoCoreError, ClientError
import os

app = FastAPI(title="Face Compare Service (Direct Upload)")

# Initialize Rekognition client
rekognition = boto3.client(
    "rekognition",
    region_name=os.getenv("AWS_REGION", "us-east-1")
)

@app.post("/compare_faces")
async def compare_faces(
    source_image: UploadFile = File(...),
    target_image: UploadFile = File(...),
    similarity_threshold: float = 90.0
):
    try:
        # Read uploaded files into memory
        source_bytes = await source_image.read()
        target_bytes = await target_image.read()

        response = rekognition.compare_faces(
            SourceImage={'Bytes': source_bytes},
            TargetImage={'Bytes': target_bytes},
            SimilarityThreshold=similarity_threshold
        )

        return {
            "matches": [
                {
                    "similarity": m["Similarity"],
                    "bounding_box": m["Face"]["BoundingBox"]
                }
                for m in response.get("FaceMatches", [])
            ],
            "unmatched_faces": len(response.get("UnmatchedFaces", []))
        }

    except (BotoCoreError, ClientError) as e:
        raise HTTPException(status_code=500, detail=str(e))
