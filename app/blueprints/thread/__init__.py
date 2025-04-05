from flask import Blueprint

thread_bp = Blueprint('thread', __name__, url_prefix='/thread')

from . import routes  # Import routes after defining the blueprint