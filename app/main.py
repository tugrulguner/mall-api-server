'''
File: main.py
File Created: 2020-07-25
Author: Parijat Khan (khanparijat@gmail.com)
-----
Copyright 2020 Parijat Khan
'''
from flask import Flask 
  
app = Flask(__name__) 
  
@app.route("/") 
def home_view(): 
    return "<h1>Welcome to A-Teams</h1>"