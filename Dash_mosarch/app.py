from PIL import Image, ImageDraw
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
ort_session = rt.InferenceSession('best.onnx')

class_names = ['Modern', 'Palaty', 'Brutalism', 'Constructivism', 'Socialist_classicism', 'Industrial_XIX', 'Panelka', 'XXI_century', 'Classicism', 'Church', 'Fortification']

app.layout = html.Div([
    html.Header(
            "This is app for recognition of architectural style of building by photo",
            style={"font-size": "30px", "textAlign": "center", 'font-family': 'Helvetica'},
        ),
    dcc.Upload(
        id='upload-image',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select File (the image will be croped automaticaly for recognition)')
        ]),
        style={
            'width': '60%',
            'height': '80px',
            'lineHeight': '80px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px',
            'font-family': 'Helvetica'
        },
        # Allow multiple files to be uploaded
        multiple=False
    ),
    html.Div(id='output-image-upload'),
    html.Div(id='output-prediction-text'),
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

# functions for onnx segmentation

def preprocess_image(image_path):
  #  image = Image.open(image_path)
    image = image_path.resize((640, 640))
    image_array = np.array(image)
    image_array = image_array.transpose(2, 0, 1)  # HWC to CHW
    image_array = image_array / 255.0 
    image_array = np.expand_dims(image_array, axis=0)
    image_array = image_array.astype(np.float32)
    return image_array

def run_inference(ort_session, image_path):
    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name
    
    # Preprocess the image
    input_data = preprocess_image(image_path)
    
    # Run inference
    outputs = ort_session.run([output_name], {input_name: input_data})[0]
    return outputs

def process_output(output, conf_threshold=0.1, iou_threshold=0.45):
    boxes = []
    for prediction in output[0]:
        confidence = prediction[4]
        if confidence >= conf_threshold:
            x, y, w, h = prediction[:4]
            class_id = np.argmax(prediction[5:])
            boxes.append([x, y, w, h, confidence, class_id])
    
    # Apply Non-Maximum Suppression (NMS)
    boxes = np.array(boxes)
    indices = nms(boxes[:, :4], boxes[:, 4], iou_threshold)
    return boxes[indices]

def nms(bboxes, scores, iou_threshold):
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = x1 + bboxes[:, 2]
    y2 = y1 + bboxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]

    return keep

def draw_boxes(image_path, boxes, class_names):
    #image = Image.open(image_path)
    image = image_path.resize((640, 640))
    draw = ImageDraw.Draw(image)

    for box in boxes[:1]:
        x, y, w, h, confidence, class_id = box
        x1, y1 = int(x - w/2), int(y - h/2)
        x2, y2 = int(x + w/2), int(y + h/2)
        
        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        
        # Draw label
        label = f"{class_names[int(class_id)]}: {confidence:.2f}"
        draw.text((x1, y1 - 10), label, fill="red")
    
    return image

def crop_image(image_path, box):
   # image = Image.open(image_path)
    image = image_path.resize((640, 640))
    x, y, w, h = box[:4]
    x1, y1 = int(x - w/2), int(y - h/2)
    x2, y2 = int(x + w/2), int(y + h/2)
    cropped_image = image.crop((x1, y1, x2, y2))
    return cropped_image

def resize_image_pil(img):
    # Convert the image string to PIL Image
    img = Image.open(io.BytesIO(base64.b64decode(img.split(",")[1])))  #.convert('L')
    img = img.convert("RGB")
    try:
        cropped_img = crop_image(img, process_output(run_inference(ort_session, img))[0])
        imag_bbox = cropped_img
        imag_bbox = draw_boxes(img, process_output(run_inference(ort_session, img)), 'building')
        crop_fact = 'Successful detection. Returning cropped image.'
    except:
        img = img.resize((640, 640))
        cropped_img = img
        imag_bbox = img
        crop_fact = 'There is no building detected, probably style in table below. Returning uncropped image.'
   
    return cropped_img, imag_bbox, crop_fact

def process_and_predict(image_string):
    cropped_img, imag_bbox, crop_fact = resize_image_pil(image_string)
    
    # Load pre-trained model
    #model = load_model(export_path)  
    
    # Perform prediction using the model
    img_resized = cropped_img.resize((256, 256))
    img_resized = img_resized.convert('L')
    img_resized = img_resized.convert("RGB")
    img_resized = np.array(img_resized)
    a = np.expand_dims(img_resized / 255.0, axis=0).astype(np.float32)
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

    return class_idx, cropped_img, imag_bbox, crop_fact # Return the predicted class index

def pil_image_to_base64(img):
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"

@app.callback(Output('output-image-upload', 'children'),
              Output('output-table-container', 'children'),
              Output('output-prediction-text', 'children'),
              Input('upload-image', 'contents'), prevent_initial_call=True)

def update_output(contents):
    if contents is not None:
        # Process image and get prediction
        df, cropped_img, imag_bbox, crop_fact = process_and_predict(contents)
        
        # Convert the cropped image to base64
        cropped_img_base64 = pil_image_to_base64(imag_bbox)
        
        # Display cropped image
        children_img = html.Div([
            html.Img(src=cropped_img_base64, style={'width': '320px', 'height': '320px'}),
            html.Hr(),
            html.Div(id='output-prediction')
        ])
        
        # Create a table for the prediction results
        table = dash_table.DataTable(
            id='table',
            columns=[{'name': col, 'id': col} for col in df.columns],
            data=df.to_dict('records'),
                style_table={'textAlign': 'left', 'height': '150px', 'overflowY': 'auto', 'width': '350px', 'font-family': 'Helvetica'},
                style_cell={'textAlign': 'left', 'font-family': 'Helvetica'}
        )
        text_phrase = html.Div(crop_fact, style={'font-size': '20px', 'color': 'black', 'font-family': 'Helvetica'})
        # Return the updated image and prediction table
        return children_img, table, text_phrase

if __name__ == '__main__':
    app.run_server(debug=False, host='0.0.0.0', port=8080)