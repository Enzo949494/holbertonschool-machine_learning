#!/usr/bin/env python3
import tensorflow_datasets as tfds
import os

manual_dir = 'my_manual_downloads'
download_config = tfds.download.DownloadConfig(manual_dir=manual_dir)

data, info = tfds.load(
    'ted_hrlr_translate/pt_to_en',
    with_info=True,
    as_supervised=True,
    download_and_prepare_kwargs={'download_config': download_config}
)

print("Dataset préparé !")
print(f"Splits: {info.splits}")

for pt, en in data['train'].take(1):
    print(pt.numpy().decode('utf-8'))
    print(en.numpy().decode('utf-8'))
