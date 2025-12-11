# This file is where the whole website runs
from flask import Flask
from .config import Config # the . means that I am importing a file that is in same package
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

app = Flask(__name__)
app.config.from_object(Config) # this is importing the variables from config.py file Config Class to put in the app.config

db = SQLAlchemy(app)
migrate = Migrate(app, db)

from . import models
from . import routes