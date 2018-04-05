import time
from threading import Lock, Thread
from flask import Flask, url_for, request, Response
from imgdb import init_imgdb
import json

app = Flask(__name__)

def get_image_batch(n_images):
    images = []
    while que.qsize() < n_images:
        time.sleep(1)
    for n in range(n_images):
        img = que.get()
        img = img.tolist()
        images.append(img)
    return images

@app.route('/get/dataset/<count>', methods = ['GET'])
def api_get_batch(count):
    count = int(count)
    batch = get_image_batch(count)
    batch = json.dumps(batch)
    resp = Response(batch, status=200, mimetype='application/json')
    return resp

if __name__ == "__main__":
    global loader, scouts, que, que_thread
    (loader, scouts, que) = init_imgdb(path="./quilts", n_scouts=10)
    app.run(host="0.0.0.0", port=10999)
