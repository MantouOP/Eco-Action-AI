import json
import boto3
from flask import Flask, request, Response, stream_with_context

app = Flask(__name__)
bedrock = boto3.client(service_name='bedrock-runtime', region_name='ap-southeast-1')
MODEL_ID = 'global.anthropic.claude-haiku-4-5-20251001-v1:0'

def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
    return response

@app.route('/', methods=['OPTIONS'])
def options_handler():
    return add_cors_headers(Response(status=200))

@app.route('/', methods=['POST'])
def generate():
    data = request.json or {}
    activity = data.get('activity', '')
    location = data.get('location', '')
    carbon_analysis = data.get('carbon_analysis', '')
    
    prompt = f"""
    Based on the following activity and its estimated carbon footprint analysis, provide 2-3 specific, actionable green alternatives that the user could do instead to reduce their impact.
    Make the suggestions practical for someone living in or near {location}.
    
    Activity: {activity}
    Carbon Analysis: {carbon_analysis}
    
    Format your response in Markdown. Use bullet points and bold text for emphasis.
    """
    
    payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1000,
        "temperature": 0.4,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}]
            }
        ]
    }
    
    def generate_stream():
        try:
            response = bedrock.invoke_model_with_response_stream(
                modelId=MODEL_ID,
                body=json.dumps(payload),
                contentType='application/json',
                accept='application/json'
            )
            for event in response.get('body'):
                chunk = json.loads(event['chunk']['bytes'])
                if chunk['type'] == 'content_block_delta':
                    yield chunk['delta'].get('text', '')
        except Exception as e:
            yield f"Error calling Bedrock: {str(e)}"
            
    response = Response(stream_with_context(generate_stream()), content_type='text/plain; charset=utf-8')
    return add_cors_headers(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
