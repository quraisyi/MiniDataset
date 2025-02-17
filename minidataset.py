import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt
from collections import Counter
import nltk
from nltk.corpus import stopwords

# download daftar stopword
nltk.download('stopwords')

df = pd.read_csv('dataminijudul.csv')

# fungsi preprocessing data
def preprocess_text(text):
    # mengubah teks menjadi huruf kecil
    text = text.lower()
    # menghapus angka dan tanda baca
    text = re.sub(r'[0-9]', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    # menghapus spasi ekstra
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# menerapkan preprocessing data ke kolom judul
df['cleaned'] = df['judul'].apply(preprocess_text)

# tokenisasi
df['tokens'] = df['cleaned'].apply(lambda x: x.split())

# mengambil daftar stopword
stop_words = set(stopwords.words('english'))

# menghapus stopword dari token
df['tokens'] = df['tokens'].apply(lambda x: [word for word in x if word not in stop_words])

# menghitung frekuensi kata
all_words = [word for tokens in df['tokens'] for word in tokens]
word_freq = Counter(all_words)

print(df[['judul', 'cleaned', 'tokens']].to_string(index=False))

# visualisasi frekuensi kata
most_common_words = word_freq.most_common(10)
words, counts = zip(*most_common_words)

plt.figure(figsize=(10, 5))
plt.bar(words, counts)
plt.title('Frekuensi Kata Teratas')
plt.xlabel('Kata')
plt.ylabel('Frekuensi')
plt.xticks(rotation=45)
plt.show()
