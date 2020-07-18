import tensorflow as tf

def encode_sentence(tokenizer, s, tokenized=False):
    if tokenized:
        tokens = s
    else:
        tokens = list(tokenizer.tokenize(s.numpy()))
    tokens.append('[SEP]')
    return tokenizer.convert_tokens_to_ids(tokens)

def get_bert_inputs(tokenizer, seq1, seq2=None, tokenized=False):
    # encode sentence and preppend CLS, and padding (tf.ragged)
    sentence1 = tf.ragged.constant([encode_sentence(tokenizer, s, tokenized) for s in seq1])
    cls = [tokenizer.convert_tokens_to_ids(['[CLS]'])] * seq1.shape[0]

    # segment mask
    input_seg_cls = tf.zeros_like(cls)
    input_seg1 = tf.zeros_like(sentence1)

    if seq2 != None:
        sentence2 = tf.ragged.constant([encode_sentence(tokenizer, s, tokenized) for s in seq2])
        input_words = tf.concat([cls, sentence1, sentence2], axis=-1)

        input_seg2 = tf.ones_like(sentence2)
        input_segment = tf.concat([input_seg_cls, input_seg1, input_seg2], axis=-1)
    else:
        input_words = tf.concat([cls, sentence1], axis=-1)
        input_segment = tf.concat([input_seg_cls, input_seg1], axis=-1)

    # attention mask
    input_mask = tf.ones_like(input_words)

    return input_words.to_tensor(), input_mask.to_tensor(), input_segment.to_tensor()