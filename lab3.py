import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

plt.rcParams['figure.figsize'] = (12, 6)
sns.set_style('whitegrid')

url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
data = pd.read_csv(url, sep='\t', names=['Class', 'Text'])

print(f"Размер датасета: {data.shape[0]} сообщений")
print(f"Распределение классов:\n{data['Class'].value_counts()}")

# Визуализация распределения классов
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

data['Class'].value_counts().plot.pie(
    ax=axes[0],
    autopct='%1.1f%%',
    colors=['skyblue', 'lightcoral'],
    labels=['Ham (не спам)', 'Spam (спам)'],
    startangle=90
)
axes[0].set_title('Распределение категорий')
axes[0].set_ylabel('')

data['Class'].value_counts().plot.bar(
    ax=axes[1],
    color=['skyblue', 'lightcoral'],
    edgecolor='black'
)
axes[1].set_title('Количество сообщений по категориям')
axes[1].set_xlabel('Категория')
axes[1].set_ylabel('Количество')
axes[1].tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.show()

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    data['Text'],
    data['Class'],
    test_size=0.2,
    random_state=42,
    stratify=data['Class']
)

print(f"Обучающая выборка: {len(X_train)} сообщений")
print(f"Тестовая выборка:  {len(X_test)} сообщений")

# Векторизация текста (CountVectorizer сам делает lowercase и токенизацию)
vectorizer = CountVectorizer()

X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

print(f"Размер словаря: {len(vectorizer.get_feature_names_out())} слов")

# Преобразование в DataFrame для анализа частот слов
X_train_df = pd.DataFrame(
    X_train_vectorized.toarray(),
    columns=vectorizer.get_feature_names_out(),
    index=X_train.index
)

spam_words = X_train_df[y_train == 'spam'].sum()
ham_words = X_train_df[y_train == 'ham'].sum()

top_spam_words = spam_words.sort_values(ascending=False).head(10)
top_ham_words = ham_words.sort_values(ascending=False).head(10)

# Визуализация топ-слов
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

top_spam_words.plot(kind='barh', ax=axes[0], color='lightcoral', edgecolor='black')
axes[0].set_title('Топ-10 слов в спам-сообщениях')
axes[0].set_xlabel('Частота')
axes[0].invert_yaxis()

top_ham_words.plot(kind='barh', ax=axes[1], color='skyblue', edgecolor='black')
axes[1].set_title('Топ-10 слов в не спам-сообщениях')
axes[1].set_xlabel('Частота')
axes[1].invert_yaxis()

plt.tight_layout()
plt.show()


# Формула: c_map = argmax_c [ log(D_c / D) + Σ_i log( (W_ic + 1) / (|V| + Σ_i' W_i'c) ) ]
def predict_naive_bayes(text):
    if not isinstance(text, str) or text.strip() == "":
        return None, 0.0, 0.0

    text_vectorized = vectorizer.transform([text]).toarray()[0]

    # Априорные вероятности P(c) = D_c / D
    spam_count = (y_train == 'spam').sum()
    ham_count = (y_train == 'ham').sum()
    total_count = len(y_train)

    p_spam = spam_count / total_count
    p_ham = ham_count / total_count

    # Условные вероятности со сглаживанием Лапласа
    vocab_size = len(vectorizer.get_feature_names_out())

    p_word_given_spam = (spam_words + 1) / (spam_words.sum() + vocab_size)
    p_word_given_ham = (ham_words + 1) / (ham_words.sum() + vocab_size)

    # Логарифмы вероятностей (для предотвращения underflow)
    log_p_spam = np.log(p_spam) + np.sum(np.log(p_word_given_spam) * text_vectorized)
    log_p_ham = np.log(p_ham) + np.sum(np.log(p_word_given_ham) * text_vectorized)

    max_log = max(log_p_spam, log_p_ham)

    # Преобразование обратно в вероятности
    p_spam_given_text = np.exp(log_p_spam - max_log)
    p_ham_given_text = np.exp(log_p_ham - max_log)

    # Нормализация
    total = p_spam_given_text + p_ham_given_text
    p_spam_given_text /= total
    p_ham_given_text /= total

    prediction = 'spam' if p_spam_given_text > p_ham_given_text else 'ham'
    return prediction, p_spam_given_text, p_ham_given_text


# Проверка функции на одном примере
test_text = "Win a free iPhone now"
prediction, p_spam, p_ham = predict_naive_bayes(test_text)
print(f"Текст: '{test_text}'")
print(f"Предсказание: {prediction}")
print(f"Вероятность спам:    {p_spam:.4f}")
print(f"Вероятность не спам: {p_ham:.4f}\n")

# Применение классификатора к тестовой выборке
y_pred = []
for text in X_test:
    prediction, _, _ = predict_naive_bayes(text)
    y_pred.append(prediction)

y_pred = np.array(y_pred)
print(f"Классификация завершена. Обработано {len(y_pred)} сообщений.\n")

# Оценка модели
accuracy = accuracy_score(y_test, y_pred)

print("=" * 60)
print("ОЦЕНКА МОДЕЛИ")
print("=" * 60)
print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n")
print("Отчет о классификации:")
print(classification_report(y_test, y_pred))

# Матрица ошибок
conf_matrix = confusion_matrix(y_test, y_pred, labels=['ham', 'spam'])
print(f"\nМатрица ошибок (строки — истина, столбцы — предсказание):")
print(f"             Предсказано: ham   spam")
print(f"  Истина ham:  {conf_matrix[0][0]:5d}  {conf_matrix[0][1]:5d}")
print(f"  Истина spam: {conf_matrix[1][0]:5d}  {conf_matrix[1][1]:5d}")

# Визуализация матрицы ошибок
plt.figure(figsize=(8, 6))

sns.heatmap(
    conf_matrix,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Ham', 'Spam'],
    yticklabels=['Ham', 'Spam'],
    cbar_kws={'label': 'Количество'}
)

plt.title('Матрица ошибок (Confusion Matrix)')
plt.xlabel('Предсказанный класс')
plt.ylabel('Истинный класс')
plt.tight_layout()
plt.show()

# Проверка на новых сообщениях
test_messages = [
    "Win a free iPhone now! Click here to claim your prize!",
    "Hey, are we still meeting for lunch tomorrow?",
    "Congratulations! You've won $1000000! Call now!",
    "Can you send me the report by Friday?",
    "URGENT: Your account has been compromised. Click here to verify.",
    "Please call our customer service representative on 0800 169 6031",
    "Good morning, how are you doing today?",
    "FREE entry into our weekly competition just text WIN to 80086"
]

print("\n" + "=" * 60)
print("ПРОВЕРКА НА НОВЫХ СООБЩЕНИЯХ")
print("=" * 60)

for msg in test_messages:
    prediction, p_spam, p_ham = predict_naive_bayes(msg)
    marker = "[СПАМ]" if prediction == 'spam' else "[НЕ СПАМ]"
    print(f"\n{marker}  (p_spam={p_spam:.4f}, p_ham={p_ham:.4f})")
    print(f"  Текст: {msg}")