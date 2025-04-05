from flask import Flask
from config import config
from app.blueprints.auth import auth_bp
from app.blueprints.index import index_bp
import mongoengine

mongo = mongoengine

def createApp(configName='default'):
    """Application factory function"""
    app = Flask(__name__)

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

    return app