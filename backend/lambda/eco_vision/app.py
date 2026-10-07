import json
import boto3
import base64

bedrock = boto3.client(service_name='bedrock-runtime', region_name='ap-southeast-1')
MODEL_ID = 'amazon.nova-canvas-v1:0'

def add_cors_headers():
    return {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Content-Type',
        'Access-Control-Allow-Methods': 'POST, OPTIONS'
    }

def handler(event, context):
    if event.get('httpMethod') == 'OPTIONS':
        return {
            'statusCode': 200,
            'headers': add_cors_headers(),
            'body': ''
        }
        
    try:
        body = json.loads(event.get('body', '{}'))
    except:
        body = {}
        
    alternatives = body.get('alternatives', '')
    
    prompt = f"A beautiful, inspiring, photorealistic visualization of a sustainable future implementing these green alternatives: {alternatives}. High quality, cinematic lighting."
    
    payload = {
        "taskType": "TEXT_IMAGE",
        "textToImageParams": {
            "text": prompt
        },
        "imageGenerationConfig": {
            "numberOfImages": 1,
            "quality": "standard",
            "cfgScale": 8.0,
            "height": 1024,
            "width": 1024
        }
    }
    
    try:
        response = bedrock.invoke_model(
            modelId=MODEL_ID,
            body=json.dumps(payload),
            contentType='application/json',
            accept='application/json'
        )
        response_body = json.loads(response.get('body').read())
        # Nova Canvas returns images in 'images' array as base64 strings
        image_base64 = response_body.get('images', [])[0]
        
        return {
            'statusCode': 200,
            'headers': add_cors_headers(),
            'body': json.dumps({'image': image_base64})
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'headers': add_cors_headers(),
            'body': json.dumps({'error': str(e)})
        }
