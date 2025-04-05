from flask import jsonify
from . import index_bp  # Import the blueprint


@index_bp.route('/')
def index():
    return jsonify("Hello World")

