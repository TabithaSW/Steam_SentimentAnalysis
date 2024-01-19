from app import app  # Import the 'app' instance created in '__init__.py'

from flask import render_template  # Import 'render_template' function for rendering HTML templates

@app.route('/')  # The '@app.route' decorator tells Flask what URL should trigger the function below
def home():
    return render_template('home.html')  # Render the 'home.html' template when the home page ('/') is accessed
