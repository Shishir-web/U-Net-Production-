from flask import Flask, request, jsonify, make_response
import onnxruntime as ort
import numpy as np
from PIL import Image
import io, base64

app = Flask(__name__)
session = ort.InferenceSession('models/unet.onnx')

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CLASSES = ['urban', 'agriculture', 'rangeland',
           'forest', 'water', 'barren', 'unknown']
COLORS = [
    [0,255,255], [255,255,0], [255,0,255],
    [0,255,0],   [0,0,255],   [255,255,255], [0,0,0]
]

def corsify(r):
    r.headers['Access-Control-Allow-Origin'] = '*'
    r.headers['Access-Control-Allow-Headers'] = '*'
    r.headers['Access-Control-Allow-Methods'] = '*'
    return r

def preprocess(b64):
    img = Image.open(io.BytesIO(base64.b64decode(b64))).convert('RGB')
    img = img.resize((256, 256), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    return arr.transpose(2,0,1).reshape(1, 3, 256, 256)

def mask_to_b64(mask):
    h, w  = mask.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, rgb in enumerate(COLORS):
        color[mask == cls] = rgb
    img = Image.fromarray(color)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')

@app.route('/segment', methods=['POST', 'OPTIONS'])
def segment():
    if request.method == 'OPTIONS':
        return corsify(make_response('', 204))
    try:
        data  = request.json.get('image')
        inp   = preprocess(data)
        out   = session.run(['output'], {'input': inp})[0]
        mask  = out[0].argmax(axis=0).astype(np.uint8)
        unique, counts = np.unique(mask, return_counts=True)
        distribution   = {
            CLASSES[int(u)]: round(int(c) / mask.size * 100, 1)
            for u, c in zip(unique, counts)
        }
        print(f"Segmentation: {distribution}")
        return corsify(jsonify({
            'mask':         mask_to_b64(mask),
            'distribution': distribution
        }))
    except Exception as e:
        print(f"Error: {e}")
        return corsify(jsonify({'error': str(e)})), 500

if __name__ == '__main__':
    app.run(port=5002, debug=False)