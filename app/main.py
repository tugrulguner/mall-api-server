'''
File: main.py
File Created: 2020-07-25
Author: Parijat Khan (khanparijat@gmail.com)
-----
Copyright 2020 Parijat Khan
'''
from flask import Flask, request, abort
from tinydb import TinyDB, Query
from tinydb.operations import delete, increment, decrement, add, subtract, set
import json
  
app = Flask(__name__) 
db = TinyDB('db.json')
density_table = db.table('density_table')
mask_table = db.table('mask_table')
  
@app.route("/") 
def home_view(): 
    return "<h1>Welcome to A-Teams</h1><h5>API server for Edge Devices</h5>"

@app.route("/density", methods=['GET', 'POST'])
def density_view():
    if request.method == 'POST':
        if request.is_json:
            content = request.json
            if ('x' in content) and ('y' in content) and ('in' in content) and ('out' in content):
                x = content['x']
                y = content['y']
                inwards = content['in']
                outwards = content['out']
                Camera = Query()
                cam = density_table.get(Camera.x == x and Camera.y == y)
                if cam is None:
                    density_table.insert({'x': x, 'y': y, 'count': inwards - outwards})
                else:
                    density_table.update(add('count', (inwards - outwards)), Camera.x == x and Camera.y == y)
                
                return json.dumps({'success':True}), 200, {'ContentType':'application/json'} 
        abort(400) 
    else:
        return json.dumps(density_table.all()), 200, {'ContentType':'application/json'} 

@app.route("/cleardensity", methods=['POST'])
def clear_density():
    density_table.truncate()
    return json.dumps({'success':True}), 200, {'ContentType':'application/json'} 

@app.route("/mask", methods=['GET', 'POST'])
def mask_view():
    if request.method == 'POST':
        if request.is_json:
            content = request.json
            if ('mask' in content) and ('nomask' in content):
                mask_val = content['mask']
                no_mask_val = content['nomask']
                
                Mask = Query()
                mask = mask_table.get(Mask.type == 'with')
                no_mask = mask_table.get(Mask.type == 'without')
                if mask is None:
                    mask_table.insert({'type': 'with', 'count': 0})
                if no_mask is None:
                    mask_table.insert({'type': 'without', 'count': 0})

                mask_table.update(add('count', mask_val), Mask.type == 'with')
                mask_table.update(add('count', no_mask_val), Mask.type == 'without')
                
                return json.dumps({'success':True}), 200, {'ContentType':'application/json'} 
        
        abort(400) 
    else:
        return json.dumps(mask_table.all()), 200, {'ContentType':'application/json'}

@app.route("/clearmask", methods=['POST'])
def clear_mask():
    mask_table.truncate()
    return json.dumps({'success':True}), 200, {'ContentType':'application/json'} 