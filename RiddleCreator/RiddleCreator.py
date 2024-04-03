import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Предположим, у нас есть список загадок вместе с ответами
riddles_with_answers = [
    "Что может путешествовать по миру, оставаясь в углу? марка",
    "Что умеет говорить, не имея языка? письмо",
    "Что растет вверх ногами? гвоздика",
    "Какой конь не ест овс? шахматный"
]

# Токенизация текста
tokenizer = Tokenizer()
tokenizer.fit_on_texts(riddles_with_answers)
total_words = len(tokenizer.word_index) + 1

# Создание последовательностей
input_sequences = []
for line in riddles_with_answers:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

# Дополнение последовательностей
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

# Создание предикторов и метки
X, labels = input_sequences[:,:-1],input_sequences[:,-1]
y = tf.keras.utils.to_categorical(labels, num_classes=total_words)

# Построение модели
model = Sequential([
    Embedding(total_words, 100, input_length=max_sequence_len-1),
    LSTM(150, return_sequences=True),
    LSTM(100),
    Dense(total_words, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Обучение модели
model.fit(X, y, epochs=100, verbose=1)

# Генерация новой загадки с ответом
def generate_riddle(seed_text, next_words=50):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predictions = model.predict(token_list, verbose=0)
        predicted = np.argmax(predictions, axis=-1)[0]
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
        if output_word == "?":  # Предполагаем, что вопрос заканчивается знаком вопроса
            break
    return seed_text

# Пример генерации
print(generate_riddle("Что может"))

