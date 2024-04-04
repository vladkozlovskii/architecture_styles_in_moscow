#!/usr/bin/env python
# coding: utf-8

# In[31]:


from PIL import Image
import dash
from dash import Dash, dcc, html, Input, Output, ctx
import io
import base64
import numpy as np
import pandas as pd
from dash import dash_table
import onnxruntime as rt

app = dash.Dash(__name__)
model = rt.InferenceSession('model.onnx')
class_names = ['Brutalism', 'Church', 'Classicism', 'Constructivism', 'Fortification', 'Industrial_XIX', 'Modern', 'Palaty', 'Panelka', 'Socialist_classicism', 'XXI_century']

app.layout = html.Div([
    html.Header(
            "This is app for recognition of architectural style of building by photo",
            style={"font-size": "30px", "textAlign": "center"},
        ),
    dcc.Upload(
        id='upload-image',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files (it would be better to crop the image along the boundaries of the building)')
        ]),
        style={
            'width': '60%',
            'height': '80px',
            'lineHeight': '80px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=False
    ),
    html.Div(id='output-image-upload'),
    html.Div(id='output-table-container')
    #html.Div(id='output-prediction', style={'width': '50%', 'height': 'auto'})
    #html.Div(class_label, style={"font-size": "20px"}, id='output-prediction')
])

def parse_contents(contents):
    return html.Div([
        html.Img(src=contents, id='img', style={'width': '50%', 'height': 'auto'}),
        html.Hr(),
        html.Div(id='output-prediction')
    ])

def resize_image_pil(img):
    # Convert the image string to PIL Image
    img = Image.open(io.BytesIO(base64.b64decode(img.split(",")[1]))).convert('L')
    img = img.convert("RGB")
    img = img.resize((256, 256))
   
    return np.array(img)

def process_and_predict(image_string):
    img_resized = resize_image_pil(image_string)
    
    # Load pre-trained model
    #model = load_model(export_path)  
    
    # Perform prediction using the model
    a = np.expand_dims(img_resized / 255.0, axis=0)
    input_name = model.get_inputs()[0].name
    output_name = model.get_outputs()[0].name
    pred = model.run([output_name], {input_name: a.astype(np.float32)})[0]
    #prediction = model.predict(np.expand_dims(img_resized / 255.0, axis=0))
    
    class_name = []
    probability = []

    for i in np.argsort(pred, axis=1)[:, -3:][0][::-1]:
        class_name.append(class_names[i]) 
        probability.append(str(np.round(pred[0][i],2))) 

    class_idx = pd.DataFrame({'Style': class_name, 'Probability': probability})    

    return class_idx  # Return the predicted class index

@app.callback(Output('output-image-upload', 'children'),
              Output('output-table-container', 'children'),
              Input('upload-image', 'contents'), prevent_initial_call=True)

def update_output(contents):
    if contents is not None:
        children_img = parse_contents(contents)
        df = process_and_predict(contents)
               
        table = dash_table.DataTable(
            id='table',
            columns=[
                {'name': col, 'id': col} for col in df.columns
            ],
            data=df.to_dict('records'),
            style_table={'textAlign': 'left','height': '150px', 'overflowY': 'auto', 'width':'350px'},
            style_cell={'textAlign': 'left'}#'minWidth': 125, 'maxWidth': 125, 'width': 125}
        )
        
        return children_img, table

if __name__ == '__main__':
    app.run_server(debug=False, host='0.0.0.0', port=8080)


# In[ ]:




