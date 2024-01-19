from flask import Flask  # Import the Flask class from the 'flask' package

app = Flask(__name__)  # Create an instance of the Flask class. '__name__' is a Python special variable which gives Python the name of the module in which it is used.

from app import routes  # Import the 'routes' module, define in the app directory. 
# This import is at the bottom to avoid circular dependencies, as 'routes' will need to import the 'app' variable defined above.
