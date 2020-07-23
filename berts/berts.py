import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_hub as hub

def BertClassificationModel(
    pretrain_url,
    classes,
    return_sequences=False,
    max_seq_length=None,
    dropout_rate=0.1,
    train_bert=True):

    # inputs
    input_words_seq = keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name='input_words_seq')
    input_attention_mask = keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name='input_attention_mask')
    input_segment_mask = keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name='input_segment_mask')

    # pre-trained bert
    bert_layers = hub.KerasLayer(pretrain_url, trainable=train_bert)
    pooled_output, seq_outputs = bert_layers([input_words_seq, input_attention_mask, input_segment_mask])

    # classfication layer
    if return_sequences:
        classification_input = seq_outputs
    else:
        classification_input = pooled_output
    output = keras.layers.Dropout(dropout_rate)(classification_input)
    if classes <= 2: # binary classfication
        output = keras.layers.Dense(1, activation='sigmoid')(output)
    else: # categorical classfication
        output = keras.layers.Dense(classes, activation='softmax')(output)

    model = keras.models.Model(inputs=[input_words_seq, input_attention_mask, input_segment_mask], outputs=output)

    return model, bert_layers

def BertEQAModel(
    pretrain_url,
    return_cls=False,
    max_seq_length=None,
    dropout_rate=0.1,
    train_bert=True):

    # inputs
    input_words_seq = keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name='input_words_seq')
    input_attention_mask = keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name='input_attention_mask')
    input_segment_mask = keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name='input_segment_mask')

    # pre-trained bert
    bert_layers = hub.KerasLayer(pretrain_url, trainable=train_bert)
    pooled_output, seq_outputs = bert_layers([input_words_seq, input_attention_mask, input_segment_mask])

    if return_cls:
        cls_output = keras.layers.Dropout(dropout_rate)(pooled_output)
        cls_output = keras.layers.Dense(1, activation='sigmoid', name='cls')(cls_output)

    # QA layer
    seq_outputs = keras.layers.Dropout(dropout_rate)(seq_outputs)
    ans_start = WeightedLayer()(seq_outputs)
    ans_end = WeightedLayer()(seq_outputs)
    mask = tf.cast(tf.equal(input_segment_mask, 0), tf.float32)
    ans_start += (mask * -1e9)
    ans_end += (mask * -1e9)
    ans_start = keras.layers.Activation('softmax', name='ans_start')(ans_start)
    ans_end = keras.layers.Activation('softmax', name='ans_end')(ans_end)

    model_outputs = [ans_start, ans_end]
    if return_cls:
        model_outputs.append(cls_output)
    model = keras.models.Model(inputs=[input_words_seq, input_attention_mask, input_segment_mask],
                               outputs=model_outputs
                              )

    return model, bert_layers

class WeightedLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(WeightedLayer, self).__init__()
    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(1,input_shape[-1]),
                                      initializer='random_normal',
                                      trainable=True)
        super(WeightedLayer, self).build(input_shape)

    def call(self, x):
        return keras.layers.dot([self.kernel, x], axes=[1,2])
