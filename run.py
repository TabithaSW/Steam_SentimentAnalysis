from app import app
import os

if __name__ == '__main__':
    DEBUG_MODE = os.environ.get('FLASK_DEBUG', 'False') == 'True'
    app.run(debug=DEBUG_MODE)
