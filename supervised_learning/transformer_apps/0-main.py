#!/usr/bin/env python3

Dataset = __import__('0-dataset').Dataset

data = Dataset()
for pt, en in data.data_train.take(2):
    print(pt.numpy().decode('utf-8'))
    print(en.numpy().decode('utf-8'))
for pt, en in data.data_valid.take(2):
    print(pt.numpy().decode('utf-8'))
    print(en.numpy().decode('utf-8'))
print(type(data.tokenizer_pt))
print(type(data.tokenizer_en))