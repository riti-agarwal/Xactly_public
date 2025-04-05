import os
import random
import base64
from flask import jsonify, request, current_app
from . import index_bp  # Import the blueprint


@index_bp.route('/')
def index():
    return jsonify("Hello World")

@index_bp.route('/random_images', methods=['GET'])
def get_random_images_api():
    with open("small/00/00a0a32d.jpg", "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

    return jsonify({
        "images": [
            {
                "url": encoded_string,
                "quality": 85,
                "history_quality": 85,
                "trend_quality": 85
            }
        ]
    })