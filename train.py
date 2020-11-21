import tensorflow as tf
from preprocessing import load_conversations, extract_sentences_list
from tokenization import train_tokenizer, tokenize_and_filter
from build_model import transformer, loss_function, accuracy, CustomSchedule

MAX_LENGTH = 40
BATCH_SIZE = 64
BUFFER_SIZE = 20000
NUM_LAYERS = 2
D_MODEL = 256
NUM_HEADS = 8
UNITS = 512
DROPOUT = 0.1
EPOCHS = 20

person1 = input('Enter person1 name (order does NOT matter): ')
person2 = input('Enter person2 name (order does NOT matter): ')

questions, answers = load_conversations(
    extract_sentences_list(person1, person2))

tokenizer = train_tokenizer(questions, answers)

START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]
# Vocabulary size plus start and end token
VOCAB_SIZE = tokenizer.vocab_size + 2

questions, answers = tokenize_and_filter(
    questions, answers, START_TOKEN, END_TOKEN)


dataset = tf.data.Dataset.from_tensor_slices((
    {
        'inputs': questions,
        'dec_inputs': answers[:, :-1]
    },
    {
        'outputs': answers[:, 1:]
    },
))

dataset = dataset.cache()
dataset = dataset.shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE)
dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

model = transformer(
    vocab_size=VOCAB_SIZE,
    num_layers=NUM_LAYERS,
    units=UNITS,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    dropout=DROPOUT)

learning_rate = CustomSchedule(D_MODEL)


optimizer = tf.keras.optimizers.Adam(
    learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)


model.compile(optimizer=optimizer, loss=loss_function, metrics=[accuracy])


model.fit(dataset, epochs=EPOCHS)

model.save_weights('saved_weights/saved_weights00')
