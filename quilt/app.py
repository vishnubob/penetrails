import time
from threading import Lock, Thread
from flask import Flask, url_for, request, Response
from imgdb import init_imgdb
import random
import json

app = Flask(__name__)

class Pool(Thread):
    def __init__(self, que, size=10000):
        self.size = size
        self.pool = []
        self.que = que
        self.lock = Lock()
        self.tick = time.time()
        self.tt_period = 10

        super(Pool, self).__init__()
        self.daemon = True

    def loop(self):
        img = self.que.get()
        img = img.tolist()
        self.add(img)

    def run(self):
        self.log("starting")
        while 1:
            self.loop()
            self.ticktock()

    def add(self, img):
        with self.lock:
            self.pool = [img] + self.pool[:self.size - 1]

    def get(self, count):
        with self.lock:
            pool = self.pool[:]
        random.shuffle(pool)
        return pool[:count]

    def ticktock(self):
        now = time.time()
        if (now - self.tick) > self.tt_period:
            self.alarm()
            self.tick = now

    def alarm(self):
        qsize = self.que.qsize()
        msg = "psize=%d" % len(self.pool)
        self.log(msg)

    def log(self, msg):
        msg = "%s: %s" % (self.__class__.__name__, msg)
        print(msg)

@app.route('/get/dataset/<count>', methods = ['GET'])
def api_get_batch(count):
    count = int(count)
    batch = pool.get(count)
    batch = json.dumps(batch)
    resp = Response(batch, status=200, mimetype='application/json')
    return resp

if __name__ == "__main__":
    global loader, scouts, que, que_thread, pool
    (loader, scouts, que) = init_imgdb(path="./quilts", n_scouts=10)
    pool = Pool(que)
    pool.start()
    app.run(host="0.0.0.0", port=10999)
