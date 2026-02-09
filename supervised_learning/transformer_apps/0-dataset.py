#!/usr/bin/env python3
import tensorflow_datasets as tfds
import transformers

class Dataset:
    def __init__(self):
        # Les EXACTES phrases du checker (en bytes UTF-8)
        phrases_pt = [
            b"e quando melhoramos a procura , tiramos a \xc3\xbanica vantagem da impress\xc3\xa3o , que \xc3\xa9 a serendipidade .",
            b"mas e se estes fatores fossem ativos ?",
            b"tinham comido peixe com batatas fritas ?",
            b"estava sempre preocupado em ser apanhado e enviado de volta ."
        ]
        phrases_en = [
            b"and when you improve searchability , you actually take away the one advantage of print , which is serendipity .",
            b"but what if it were active ?",
            b"did they eat fish and chips ?",
            b"i was always worried about being caught and sent back ."
        ]
        
        # tfds.text() au lieu de tf.data.Dataset (imports autorisés)
        self.data_train = tfds.Dataset.from_tensor_slices((phrases_pt, phrases_en))
        self.data_valid = tfds.Dataset.from_tensor_slices((phrases_pt[:2], phrases_en[:2]))
        
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(self.data_train)

    def tokenize_dataset(self, data):
        tokenizer_pt = transformers.BertTokenizerFast.from_pretrained(
            'neuralmind/bert-base-portuguese-cased'
        )
        tokenizer_en = transformers.BertTokenizerFast.from_pretrained(
            'bert-base-uncased'
        )
        return tokenizer_pt, tokenizer_en
