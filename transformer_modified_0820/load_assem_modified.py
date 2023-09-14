import numpy as np
import tensorflow as tf
# import ray

# ray.shutdown()
# ray.init()
import sys

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.compat.v1.Session(config=config)

physical_devices = tf.config.list_physical_devices('GPU') 
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#   try:
#     tf.config.experimental.set_virtual_device_configuration(
#         gpus[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120)])
#   except RuntimeError as e:
#     print(e)

process_num = int(sys.argv[1])

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_angles(self, position, i, d_model):
        angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return position * angles

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model=d_model,
        )

        # Compute sine and cosine components separately
        sines = tf.math.sin(angle_rads[:, 0::2])
        cosines = tf.math.cos(angle_rads[:, 1::2])

        # Interleave sine and cosine components to get the positional encoding
        pos_encoding = tf.stack([sines, cosines], axis=-1)
        pos_encoding = tf.reshape(pos_encoding, [position, d_model])
        pos_encoding = pos_encoding[tf.newaxis, ...]

        return tf.cast(pos_encoding, tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, : tf.shape(inputs)[1], :]


def scaled_dot_product_attention(query, key, value, mask):
    matmul_qk = tf.matmul(query, key, transpose_b=True)

    depth = tf.cast(tf.shape(key)[-1], tf.float32)
    logits = matmul_qk / tf.math.sqrt(depth)

    if mask is not None:
        logits += mask * -1e9

    attention_weights = tf.nn.softmax(logits, axis=-1)

    output = tf.matmul(attention_weights, value)

    return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, name="multi_head_attention"):
        super(MultiHeadAttention, self).__init__(name=name)
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.query_dense = tf.keras.layers.Dense(units=d_model)
        self.key_dense = tf.keras.layers.Dense(units=d_model)
        self.value_dense = tf.keras.layers.Dense(units=d_model)

        self.dense = tf.keras.layers.Dense(units=d_model)

    def split_heads(self, inputs, batch_size):
        inputs = tf.reshape(inputs, shape=(batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(inputs, perm=[0, 2, 1, 3])

    def call(self, inputs):
        query, key, value, mask = (
            inputs["query"],
            inputs["key"],
            inputs["value"],
            inputs["mask"],
        )
        batch_size = tf.shape(query)[0]

        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)

        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        scaled_attention, _ = scaled_dot_product_attention(query, key, value, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))

        outputs = self.dense(concat_attention)

        return outputs


def create_padding_mask(x):
    mask = tf.cast(tf.math.equal(x, 0), tf.float32)
    return mask[:, tf.newaxis, tf.newaxis, :]


def encoder_layer(dff, d_model, num_heads, dropout, name="encoder_layer"):
    inputs = tf.keras.Input(shape=(None, d_model), name="inputs")

    padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")
    # print("padding mask enc:", padding_mask)
    attention = MultiHeadAttention(d_model, num_heads, name="attention")(
        {"query": inputs, "key": inputs, "value": inputs, "mask": padding_mask}
    )

    attention = tf.keras.layers.Dropout(rate=dropout)(attention)
    attention = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs + attention)

    outputs = tf.keras.layers.Dense(units=dff, activation="relu")(attention)
    outputs = tf.keras.layers.Dense(units=d_model)(outputs)

    outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
    outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention + outputs)

    return tf.keras.Model(inputs=[inputs, padding_mask], outputs=outputs, name=name)


def encoder(vocab_size, num_layers, dff, d_model, num_heads, dropout, name="encoder"):
    inputs = tf.keras.Input(shape=(None,), name="inputs")

    padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

    embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
    embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)
    outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)

    for i in range(num_layers):
        outputs = encoder_layer(
            dff=dff,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            name="encoder_layer_{}".format(i),
        )([outputs, padding_mask])

    return tf.keras.Model(inputs=[inputs, padding_mask], outputs=outputs, name=name)


def create_look_ahead_mask(x):
    seq_len = tf.shape(x)[1]
    look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    padding_mask = create_padding_mask(x)
    return tf.maximum(look_ahead_mask, padding_mask)


def decoder_layer(dff, d_model, num_heads, dropout, name="decoder_layer"):
    inputs = tf.keras.Input(shape=(None, d_model), name="inputs")
    enc_outputs = tf.keras.Input(shape=(None, d_model), name="encoder_outputs")

    look_ahead_mask = tf.keras.Input(shape=(1, None, None), name="look_ahead_mask")
    padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

    # print("look ahead mask dec:", look_ahead_mask)
    attention1 = MultiHeadAttention(d_model, num_heads, name="attention_1")(
        inputs={
            "query": inputs,
            "key": inputs,
            "value": inputs,
            "mask": look_ahead_mask,
        }
    )

    attention1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention1 + inputs)

    # print("padding mask dec:", padding_mask)
    attention2 = MultiHeadAttention(d_model, num_heads, name="attention_2")(
        inputs={
            "query": attention1,
            "key": enc_outputs,
            "value": enc_outputs,
            "mask": padding_mask,
        }
    )

    attention2 = tf.keras.layers.Dropout(rate=dropout)(attention2)
    attention2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(
        attention2 + attention1
    )

    outputs = tf.keras.layers.Dense(units=dff, activation="relu")(attention2)
    outputs = tf.keras.layers.Dense(units=d_model)(outputs)

    outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
    outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6)(outputs + attention2)

    return tf.keras.Model(
        inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
        outputs=outputs,
        name=name,
    )


def decoder(vocab_size, num_layers, dff, d_model, num_heads, dropout, name="decoder"):
    inputs = tf.keras.Input(shape=(None,), name="inputs")
    enc_outputs = tf.keras.Input(shape=(None, d_model), name="encoder_outputs")

    look_ahead_mask = tf.keras.Input(shape=(1, None, None), name="look_ahead_mask")
    padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

    embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
    embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)
    outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)

    for i in range(num_layers):
        outputs = decoder_layer(
            dff=dff,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            name="decoder_layer_{}".format(i),
        )(inputs=[outputs, enc_outputs, look_ahead_mask, padding_mask])

    return tf.keras.Model(
        inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
        outputs=outputs,
        name=name,
    )


def transformer(
    vocab_size, num_layers, dff, d_model, num_heads, dropout, name="transformer"
):
    inputs = tf.keras.Input(shape=(None,), name="inputs")
    print("inputs", inputs)

    dec_inputs = tf.keras.Input(shape=(None,), name="dec_inputs")

    enc_padding_mask = tf.keras.layers.Lambda(
        create_padding_mask, output_shape=(1, 1, None), name="enc_padding_mask"
    )(inputs)

    look_ahead_mask = tf.keras.layers.Lambda(
        create_look_ahead_mask, output_shape=(1, None, None), name="look_ahead_mask"
    )(dec_inputs)

    dec_padding_mask = tf.keras.layers.Lambda(
        create_padding_mask, output_shape=(1, 1, None), name="dec_padding_mask"
    )(inputs)

    enc_outputs = encoder(
        vocab_size=vocab_size,
        num_layers=num_layers,
        dff=dff,
        d_model=d_model,
        num_heads=num_heads,
        dropout=dropout,
    )(inputs=[inputs, enc_padding_mask])

    dec_outputs = decoder(
        vocab_size=vocab_size,
        num_layers=num_layers,
        dff=dff,
        d_model=d_model,
        num_heads=num_heads,
        dropout=dropout,
    )(inputs=[dec_inputs, enc_outputs, look_ahead_mask, dec_padding_mask])

    outputs = tf.keras.layers.Dense(units=vocab_size, name="outputs")(dec_outputs)

    return tf.keras.Model(inputs=[inputs, dec_inputs], outputs=outputs, name=name)


def loss_function(y_true, y_pred):
    y_true = tf.reshape(y_true, shape=(-1, MAX_LENGTH - 1))

    loss = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction="none"
    )(y_true, y_pred)

    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    loss = tf.multiply(loss, mask)

    return tf.reduce_mean(loss)


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=500):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(tf.cast(step, tf.float32))
        arg2 = tf.cast(step, tf.float32) * (self.warmup_steps**-1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2) 
    
with open("opcode_set.txt", "r") as opcode_file:
    lines = opcode_file.readlines()

opcodes = list(map(lambda x: x.replace("\n", ""), lines))

op2idx = {word: i for i, word in enumerate(opcodes)}
idx2op = np.array(opcodes)

VOCAB_SIZE = len(opcodes)
START_TOKEN, END_TOKEN = VOCAB_SIZE, VOCAB_SIZE + 1

op2idx["START"] = VOCAB_SIZE
op2idx["END"] = VOCAB_SIZE + 1
idx2op = np.append(idx2op, ["START", "END"])

VOCAB_SIZE = VOCAB_SIZE + 2

import numpy as np

MAX_LENGTH = VOCAB_SIZE

def encode(code):
    return list(map(lambda x: op2idx[x], code))


def decode(idx):
    return list(map(lambda x: idx2op[x], idx))
  
NUM_LAYERS = 3
D_MODEL = 40
NUM_HEADS = 40
DFF = 64
DROPOUT = 0.1

new_model = transformer(
    vocab_size=VOCAB_SIZE,
    num_layers=NUM_LAYERS,
    dff=DFF,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    dropout=DROPOUT,
)

new_model.load_weights('transformer_modified_only_0820')

import pandas as pd
import os


df = pd.DataFrame({'errors' : [], 'sequence len' : [], 'corrects' : [],'type': [], })
    
def predict(X_test, y_test, test_meta):

    test_input = tf.keras.preprocessing.sequence.pad_sequences(
        [
            np.append(np.insert(x, 0, op2idx["START"]), op2idx["END"])
            for x in X_test
        ],
        maxlen=MAX_LENGTH,
        padding="post",
    )

    test_output = tf.keras.preprocessing.sequence.pad_sequences(
        [
            np.append(np.insert(x, 0, op2idx["START"]), op2idx["END"])
            for x in y_test
        ],
        maxlen=MAX_LENGTH,
        padding="post",
    )

    # file_meta = open('metadata.txt', 'w')
    # file_acc = open('test_accuracy.txt', 'w')
    log_file = open(f'log_m/{process_num}.txt', 'w')
    # print("log:", process_num)
    for i in range(len(X_test)):

        output = np.expand_dims([op2idx["START"]], 0)
        
        for j in range(MAX_LENGTH-1):
            predictions = new_model(
                inputs=[np.expand_dims(test_input[i], 0), output], training=False
            )
            predictions = predictions[:, -1:, :]
            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
            output = tf.concat([output, predicted_id], axis=-1)
            if tf.equal(predicted_id, END_TOKEN):
                break
        
        # print("X test", test_output[i], sep="\n")
        # print("output", output, sep="\n")

        corrected = 0
        current_length = min(int(test_meta[i][1]), len(output[0]))
        # print("X_test i", X_test[i])

        for k in range(min(current_length, MAX_LENGTH)):
            if test_output[i][k] == output[0][k]:
                corrected = corrected + 1

        df.loc[i] = [test_meta[i][2], test_meta[i][1], corrected, test_meta[i][0]]
        completely_ok = ""
        if corrected == current_length:
            completely_ok = "[완전일치]"
        else:
            completely_ok = "[부분불일치]"
        if int(test_meta[i][1]) != 40 and int(test_meta[i][1]) + 2 != len(output[0]):
            print(int(test_meta[i][1]) + 2, len(output[0]))
            completely_ok = "[길이불일치]"
       
        log_file.write(f"{test_meta[i][0]} {test_meta[i][1]} {test_meta[i][2]} {corrected} {completely_ok} {','.join(decode(output[0][1:-1]))} {','.join(decode(y_test[i]))}\n")
        # file_meta.write(f"{error_cnt} {current_length}\n")
        # file_acc.write(f"{current_acc}\n")
    # file_meta.close()
    # file_acc.close()
    log_file.close()
# X_test = X_test_total[:200000]
# y_test = y_test_total[:200000]
with open(f"datas_m/X_test_{process_num}.txt", "r") as file:
  X_test = [line.rstrip() for line in file.readlines()]
with open(f"datas_m/y_test_{process_num}.txt", "r") as file:
  y_test = [line.rstrip() for line in file.readlines()]


test_meta = [x.split(" ")[:3] for x in X_test]

X_test = [x.split(" ")[3].split(",") for x in X_test]
y_test = [x.split(" ")[4].split(",") for x in y_test]

X_test = [encode(x) for x in X_test]
y_test = [encode(y) for y in y_test]


predict(X_test, y_test, test_meta)

# df.to_csv(f'csv/test_acc_before_{process_num}.csv')

df = df.groupby(['type','sequence len','errors', 'corrects']).size()
# df = df.reset_index()

# print("after df")
# print(df)
df.to_csv(f'csv_m/predict_data_{process_num}.csv')