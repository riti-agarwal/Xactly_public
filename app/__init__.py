from flask import Flask
from config import config
from app.blueprints.auth import auth_bp
from app.blueprints.index import index_bp
from app.blueprints.thread import thread_bp

import mongoengine
from flask_cors import CORS

mongo = mongoengine

def createApp(configName='default'):
    """Application factory function"""
    app = Flask(__name__)

    # Configure CORS with strict-origin-when-cross-origin policy
    CORS(app, resources={
        r"/*": {
            "origins": ["http://localhost:3000", "http://127.0.0.1:3000"],
            "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization"],
            "expose_headers": ["Content-Type", "X-Total-Count"],
            "supports_credentials": True
        }
    })
    
    # Load configuration
    app.config.from_object(config[configName])

    mongo.connect(
        host=app.config['MONGO_URI'],
        db=app.config['MONGO_DB_NAME']
    )

    # Initialize components
    # (This would be where you initialize database, cache, etc.)
    
    app.register_blueprint(index_bp)
    app.register_blueprint(auth_bp)
    app.register_blueprint(thread_bp)


    return app