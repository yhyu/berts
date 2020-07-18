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
    output = keras.layers.Dropout(0.1)(classification_input)
    if classes <= 2: # binary classfication
        output = keras.layers.Dense(1, activation='sigmoid')(output)
    else: # categorical classfication
        output = keras.layers.Dense(classes, activation='softmax')(output)

    model = keras.models.Model(inputs=[input_words_seq, input_attention_mask, input_segment_mask], outputs=output)

    return model, bert_layers

def BertQAModel(
    pretrain_url,
    max_seq_length=None,
    question_len=None,
    dropout_rate=0.1,
    train_bert=True):

    # inputs
    input_words_seq = keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name='input_words_seq')
    input_attention_mask = keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name='input_attention_mask')
    input_segment_mask = keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name='input_segment_mask')

    # pre-trained bert
    bert_layers = hub.KerasLayer(pretrain_url, trainable=train_bert)
    pooled_output, seq_outputs = bert_layers([input_words_seq, input_attention_mask, input_segment_mask])

    # TODO: QA layer
    model = None

    return model, bert_layers