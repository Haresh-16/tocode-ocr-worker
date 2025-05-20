# text det and recog code 

import boto3, json, os, cv2, time
import numpy as np
from mmocr.apis import MMOCRInferencer

# AWS clients
sqs = boto3.client('sqs')
s3 = boto3.client('s3')
ddb = boto3.resource('dynamodb')
table = ddb.Table(os.environ['DYNAMODB_TABLE'])

# Config
queue_url = os.environ['SQS_QUEUE_URL']
bucket = os.environ['S3_BUCKET']

# Load PANNet and RobustScanner once using MMOCR
pannet = MMOCRInferencer(det='pan')
recognizer = MMOCRInferencer(rec='robustscanner')

def download_image(job_id, s3_key):
    local_path = f"/tmp/{job_id}.jpg"
    s3.download_file(bucket, s3_key, local_path)
    return local_path

def detect_best_box(image, model):
    result = model(image, return_vis=False)
    polygons = result['predictions'][0]['det_polygons']
    scores = result['predictions'][0]['det_scores']

    if not polygons or not scores:
        return None

    best_idx = np.argmax(scores)
    best_poly = np.array(polygons[best_idx])
    x1, y1 = np.min(best_poly, axis=0)
    x2, y2 = np.max(best_poly, axis=0)
    return [int(x1), int(y1), int(x2), int(y2)]

def recognize_text(image_region):
    rgb_image = cv2.cvtColor(image_region, cv2.COLOR_BGR2RGB)
    result = recognizer(rgb_image, return_vis=False)
    preds = result['predictions'][0].get('rec_texts', [])
    return preds[0] if preds else ""

def run_ocr(image_path):
    image = cv2.imread(image_path)
    box = detect_best_box(image, pannet)
    if not box:
        return {"error": "No confident detection"}
    x1, y1, x2, y2 = box
    roi = image[y1:y2, x1:x2]
    text = recognize_text(roi)
    return {"best_text": text}

while True:
    msgs = sqs.receive_message(QueueUrl=queue_url, MaxNumberOfMessages=1, WaitTimeSeconds=10)
    for msg in msgs.get('Messages', []):
        try:
            body = json.loads(msg['Body'])
            job_id = body['job_id']
            s3_key = body['s3_key']
            print(f"Processing job {job_id}")

            local_path = download_image(job_id, s3_key)
            result = run_ocr(local_path)

            # Store result in S3
            s3.put_object(
                Bucket=bucket,
                Key=f"results/{job_id}.json",
                Body=json.dumps(result)
            )

            # Store result in DynamoDB
            table.put_item(Item={
                "job_id": job_id,
                "status": "COMPLETED",
                "result": result
            })

            # Delete job from queue
            sqs.delete_message(
                QueueUrl=queue_url,
                ReceiptHandle=msg['ReceiptHandle']
            )
        except Exception as e:
            print(f"Error: {e}")
