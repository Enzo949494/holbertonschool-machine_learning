#!/usr/bin/env python3
import tensorflow_datasets as tfds

ds = tfds.load(
    'para_crawl/enpt',
    split='train',
    as_supervised=True,
    data_dir='./tensorflow_datasets'
)

ds = ds.map(lambda en, pt: (pt, en))

for pt, en in ds.take(3):
    print("\nPortugais:", pt.numpy().decode('utf-8'))
    print("Anglais:", en.numpy().decode('utf-8'))
    print("-" * 60)