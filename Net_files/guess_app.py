import pandas as pd
from flask import Flask, render_template, jsonify, request, redirect
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
import re
import os

import Hee_final

# Flask App
app = Flask(__name__)
CORS(app)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', '') or 'sqlite:///db.sqlite'
db = SQLAlchemy(app)

class Game(db.Model):
    __tablename__ = 'review_history'

    review_id = db.Column(db.Integer, primary_key=True)
    review = db.Column(db.String())
    star_human = db.Column(db.Integer)
    star_model = db.Column(db.Integer)
    star_difference = db.Column(db.Integer)

def assign_review_id():
    current_max = pd.read_sql('SELECT max(review_id) FROM review_history',db.engine).fillna(0).loc[0][0]
    new_id = current_max + 1
    return new_id