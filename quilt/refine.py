#!/usr/bin/env python
import sys
import os
import shutil
import json
import pprint
import pandas as pd
import shutil

quilt_cats = [
    'n03898633 patchwork, patchwork quilt',
    'n04033995 quilt, comforter, comfort, puff',
    'n03128427 crazy quilt',
    'n03266749 eiderdown, duvet, continental quilt',
    'n04034262 quilted bedspread'
]

# write a function that takes one or more cats
# and provides a list of paths
def get_images_from_cat(df, cat):
    idx_max_p = df.groupby(["path"], sort=False)["probability"].transform(max) == df["probability"]
    max_p = df[idx_max_p]
    #print(max_p)
    return max_p[max_p["class_id"] == cat]

def fix_path(basedir):
    def func(path):
        path = path.split("\\")
        idx = path.index("downloads")
        path = [basedir] + path[idx:]
        return os.path.join(*path)
    return func

def load_json(fn, source_path):
    with open(fn) as fh:
        data = json.load(fh)
    dat = list(data.values())
    dat = [j for i in dat for j in i]
    df = pd.DataFrame.from_records(dat)
    fixer = fix_path(source_path)
    df["path"] = df["path"].apply(fixer)
    print(df["path"].head())
    return df

def run(fn, source_path, target_path):
    source_path = os.path.abspath(source_path)
    df = load_json(fn, source_path)
    if not os.path.exists(target_path):
        os.mkdir(target_path)
    for cat in quilt_cats:
        imgs = get_images_from_cat(df, cat)
        for path in imgs["path"]:
            path = os.path.abspath(path)
            print(path)
            fn = os.path.split(path)[-1]
            target = os.path.join(target_path, fn)
            if os.path.exists(target):
                os.unlink(target)
            os.symlink(path, target)

if __name__ == "__main__":
    (json_class, source_path, target_path) = sys.argv[1:]
    run(json_class, source_path, target_path)
