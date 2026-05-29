from flask import Flask, render_template, request, jsonify
import boto3
import json
import os
from PIL import Image, ImageDraw, ImageFont
import base64
from io import BytesIO

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
RESULTS_FOLDER = 'static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Initialize AWS Rekognition client
rekognition = boto3.client('rekognition', region_name='us-east-1')


def format_user_friendly_message(labels, image_path):
    """Convert detection labels to a user-friendly message"""
    people_count = 0
    women_count = 0
    men_count = 0
    adults_count = 0
    children_count = 0
    clothing_items = {}
    vehicles = []
    other_objects = {}
    
    for label in labels:
        name = label.get('Name', '')
        confidence = label.get('Confidence', 0)
        instances = label.get('Instances', [])
        instance_count = len(instances)
        
        if name == 'Person':
            people_count = instance_count
        elif name == 'Woman':
            women_count = instance_count
        elif name == 'Man':
            men_count = instance_count
        elif name == 'Adult':
            adults_count = instance_count
        elif name == 'Child':
            children_count = instance_count
        elif name in ['Shorts', 'Shoe', 'Footwear', 'Shirt', 'Pants', 'Dress', 'Hat', 'Jacket']:
            if instance_count > 0:
                clothing_items[name] = instance_count
        elif 'scooter' in name.lower() or 'car' in name.lower() or 'bike' in name.lower() or 'bicycle' in name.lower():
            if instance_count > 0:
                vehicles.append(f"{name} ({instance_count})")
            elif confidence > 90:
                vehicles.append(name)
        elif name not in ['People', 'Female', 'Male', 'Human', 'Clothing']:
            if instance_count > 0:
                other_objects[name] = instance_count
            elif confidence > 90 and instance_count == 0:
                other_objects[name] = 'detected'
    
    return {
        'people_count': people_count,
        'women_count': women_count,
        'men_count': men_count,
        'children_count': children_count,
        'clothing_items': clothing_items,
        'vehicles': vehicles,
        'other_objects': other_objects
    }


def draw_bounding_boxes(image_path, labels):
    """Draw bounding boxes around detected objects"""
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    width, height = img.size
    
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    colors = ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'cyan', 'magenta']
    color_index = 0
    
    for label in labels:
        if label.get('Instances'):
            color = colors[color_index % len(colors)]
            color_index += 1
            
            for instance in label['Instances']:
                box = instance['BoundingBox']
                
                left = int(box['Left'] * width)
                top = int(box['Top'] * height)
                right = left + int(box['Width'] * width)
                bottom = top + int(box['Height'] * height)
                
                draw.rectangle([left, top, right, bottom], outline=color, width=3)
                
                label_text = f"{label['Name']} {instance['Confidence']:.1f}%"
                text_bbox = draw.textbbox((left + 5, top + 5), label_text, font=font)
                draw.rectangle(text_bbox, fill=color)
                draw.text((left + 5, top + 5), label_text, fill='white', font=font)
    
    base_name = os.path.basename(image_path)
    output_path = os.path.join(RESULTS_FOLDER, f"annotated_{base_name}")
    img.save(output_path)
    return output_path


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save uploaded file
        filename = file.filename
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        # Read image as bytes
        with open(filepath, 'rb') as image_file:
            image_bytes = image_file.read()
        
        # Call Rekognition API
        response = rekognition.detect_labels(
            Image={'Bytes': image_bytes},
            MaxLabels=20,
            MinConfidence=75
        )
        
        # Format results
        labels = response['Labels']
        summary = format_user_friendly_message(labels, filepath)
        
        # Draw bounding boxes
        annotated_path = draw_bounding_boxes(filepath, labels)
        
        # Prepare response
        result = {
            'success': True,
            'summary': summary,
            'labels': labels,
            'original_image': f'/static/uploads/{filename}',
            'annotated_image': f'/{annotated_path}'
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)