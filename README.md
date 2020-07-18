# BERTs
BERT applications using tensorflow

## single output classification
See [sentiment analysis](https://github.com/yhyu/berts/blob/master/sentiment.ipynb) example.
```python
bert_url = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2' # pre-trained bert on tfhub url
classes = 2 # number of classes
model, bert_layer = BertClassificationModel(bert_url, classes)
```

## sequence output classfication
See [pos tagging](https://github.com/yhyu/berts/blob/master/pos_tagging.ipynb) example.
```python
model, bert_layer = BertClassificationModel(
    bert_url,
    classes,
    return_sequences=True
)
```