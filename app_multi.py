import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html

import base64
import os
from urllib.parse import quote as urlquote
from flask import Flask, send_from_directory

from keras.preprocessing import image
from keras.models import load_model
from keras import backend as K
import numpy as np

import flask


server = Flask(__name__)
app = dash.Dash(server=server)


external_css = ["https://codepen.io/chriddyp/pen/bWLwgP.css",
                "https://cdn.rawgit.com/samisahn/dash-app-stylesheets/" +
                "eccb1a1a/dash-tektronix-350.css"]

for css in external_css:
    app.css.append_css({"external_url": css})

UPLOAD_DIRECTORY = "upload"

@server.route("/download/<path:path>")
def download(path):
    """Serve a file from the upload directory."""
    return send_from_directory(UPLOAD_DIRECTORY, path, as_attachment=True)


def save_file(name, content):
    """Decode and store a file uploaded with Plotly Dash."""
    data = content.encode("utf8").split(b";base64,")[1]
    with open(os.path.join(UPLOAD_DIRECTORY, name), "wb") as fp:
        fp.write(base64.decodebytes(data))

def remove_pic(name):
    os.remove("upload" + "/" + "{0}".format(name))


app.layout = html.Div([
    
    html.Div([
        html.Div([
            html.H1("Kissa vs Koira tunnistus konvoluutio neuroverkolla")
            ], className="banner", style={"margin-top": "10px"}),
        html.Div([
            dcc.Upload(
                id="upload-image",
                children=html.Div([
                    "Drag and Drop or ",
                    html.A("Select File")
                ]),
                style={
                    "lineHeight": "60px",
                    "borderWidth": "1px",
                    "borderStyle": "dashed",
                    "borderRadius": "5px",
                    "textAlign": "center",
                    "margin": "10px",
                    "fontSize": 20,
                    "position": "center"
                },
                multiple=True
            ),
        ], className="container"),
        
        html.Div(id="output-image-upload", style={"position": "auto"}),

    ], className="ten columns offset-by-one", style={"backgroundColor": "#f0f5f5", "height":"100%", "border-radius": "40px", "margin-top": "20px"}),
            html.Div([
            dcc.Markdown("""
_Tensorflow version: 1.14.0,_
_Keras Version: 2.2.4_
""")  
        ], className="ten columns offset-by-one", style = {"height":"5vh"}),

], className="row", style={"textAlign": "center", "backgroundColor": "#d1e0e0", "height":"100%"})


# HTML images accept base64 encoded strings in the same format
# that is supplied by the upload
def parse_contents(contents, filename, pred):
    return html.Div([
        html.H5(filename),
        html.Img(src=contents, style={"width": "300px", "height": "300px"}),
        html.Div(pred)
    ])

def prediction(name):
    file = "upload/" + "{0}".format(name)
    img = image.load_img(file, target_size = (150, 150))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis = 0)
    img /= 255.
    
    model = load_model("model2-0.07.h5")
    prediction = model.predict(img)
    K.clear_session()
    

    if prediction[0] > 0.5:
        return "Koira {:.2%} varmuudella".format(prediction[0][0])
    else:
        return "Kissa {:.2%} varmuudella".format(1-prediction[0][0])

def remove_pic(name):
    os.remove("upload" + "/" + "{0}".format(name))


@app.callback(Output('output-image-upload', 'children'),
              [Input('upload-image', 'contents')],
              [State('upload-image', 'filename')])
def update_output(list_of_contents, list_of_names):
    if list_of_contents is not None:
        for name, data in zip(list_of_names, list_of_contents):
            save_file(name, data)
            children = [
                parse_contents(c, n, prediction(name)) for c, n in
                zip(list_of_contents, list_of_names)]
            remove_pic(name)
            #children.append(pred)
        return children


if __name__ == '__main__':
    app.run_server(debug=True, port=9999)