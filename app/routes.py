# this is where we detirmine were each page goes to and what data they need to give and take
from flask import render_template, redirect, url_for
from . import app, db

@app.route("/")
def root():
    return redirect(url_for("home"))     

@app.route("/home")
def home():
    return render_template("index.html") 

@app.route("/chat")
def chat():
    return render_template("chat.html")

@app.route("/about")
def about():
    return render_template("about.html")
