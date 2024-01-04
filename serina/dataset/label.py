#!/usr/bin/env python
import json
import os

import pandas as pd
from sklearn.preprocessing import LabelEncoder


PATH = os.path.dirname(__file__) + "/label.json"

def generate():
    df = pd.read_csv( os.path.dirname(__file__) + "/../../ESC-50/meta/esc50.csv")
    categories = [v for v in df["category"].unique()]
    categories.sort()
    encoder = LabelEncoder()
    numeric_labels = encoder.fit_transform(categories)
    result = {}
    for i in range(len(categories)):
        result[categories[i]] = int(numeric_labels[i])
    with open(PATH, "w") as f:
        json.dump(result, f)


data_l2i = None
data_i2l = None


def __prepare():
    global data_l2i
    global data_i2l
    if data_l2i is None:
        with open(PATH, "r") as f:

            json_data = json.load(f)
            data_i2l = {}
            data_l2i = {}
            for k, v in json_data.items():
                data_l2i[k] = v
                data_i2l[v] = k


def index_to_label(index: int):
    __prepare()
    return data_i2l[index]


def get_categories():
    __prepare()
    return len(data_i2l)


def label_to_index(label: str):
    __prepare()
    return data_l2i[label]


if __name__ == "__main__":
    generate()
    # print(label_to_index("dog"))
    # print(index_to_label(18))
