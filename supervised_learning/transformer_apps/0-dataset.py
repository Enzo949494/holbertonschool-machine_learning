#!/usr/bin/env python3
import tensorflow_datasets as tfds
import transformers


class Dataset:
    def __init__(self):
        # Charger para_crawl/enpt et inverser pour pt->en
        data_train = tfds.load(
            'para_crawl/enpt',
            split='train',
            as_supervised=True,
            data_dir='./tensorflow_datasets'
        ).map(lambda en, pt: (pt, en))  # (en, pt) → (pt, en)

        data_valid = tfds.load(
            'para_crawl/enpt',
            split='train',  # para_crawl n'a qu'un split 'train'
            as_supervised=True,
            data_dir='./tensorflow_datasets'
        ).map(lambda en, pt: (pt, en))

        self.data_train = data_train
        self.data_valid = data_valid

        # Tokenizers BERT (inchangés)
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(self.data_train)

    def tokenize_dataset(self, data):
        tokenizer_pt = transformers.BertTokenizerFast.from_pretrained(
            'neuralmind/bert-base-portuguese-cased'
        )
        tokenizer_en = transformers.BertTokenizerFast.from_pretrained(
            'bert-base-uncased'
        )
        return tokenizer_pt, tokenizer_en
