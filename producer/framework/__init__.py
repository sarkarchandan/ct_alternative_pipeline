from flask import Flask

# Initializes container process
app: Flask = Flask(__name__)

from framework import server