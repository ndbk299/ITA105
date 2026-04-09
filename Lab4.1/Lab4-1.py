import pandas as pd                   
import numpy as np                      
import re                               
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec 

# Bài 1 Review khách sạn

print("\n===== BÀI 1 =====")

# - Nạp dữ liệu, kiểm tra missing values.
reviews = pd.read_csv(r'd:\FPT\ITA105\LMS-ITA105\Lab4\Lab4\ITA105_Lab_4_Hotel_reviews.csv', encoding='utf-8')
print(reviews.shape)
print(reviews.isnull().sum())
print(reviews.head())

# - Label Encoding hoặc One-Hot Encoding các biến categorical.
reviews_encoded = pd.get_dummies(reviews, columns=['hotel_name', 'customer_type'], drop_first=True)
print(reviews_encoded.head())


# - Tiền xử lý văn bản: lowercase, tokenization, loại bỏ stop words.
stop_words = ["là", "và", "có", "rất", "thì", "một", "những", "các", "ở"]
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', "", text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)
reviews_encoded['cleaned_review'] = reviews_encoded['review_text'].apply(clean_text)
print(reviews_encoded[['review_text', 'cleaned_review']].head())

# - Tạo TF-IDF matrix cho review khách hàng.
reviews_encoded['review_text'] = reviews_encoded['review_text'].fillna("")
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(reviews_encoded['review_text'])
print("\nTF-IDF matrix:")
print(tfidf_matrix.toarray())

# - Tạo Word2Vec embeddings cho review và tìm 5 từ gần nghĩa với “sạch sẽ”.
model = Word2Vec(sentences=[text.split() for text in reviews_encoded['cleaned_review']], vector_size=50, window=2, min_count=1)
similar_words = model.wv.most_similar("sạch", topn=5)
print("\n5 words similar to 'sạch':", similar_words)

# - Khi nào nên dùng TF-IDF và khi nào nên dùng Word2Vec?
print("\nKết luận:")
print("TF-IDF: tốt cho các mô hình truyền thống, khi bạn chỉ cần trọng số từ để phân loại hoặc tìm kiếm.")
print("Word2Vec: tốt khi bạn cần hiểu ngữ nghĩa, quan hệ giữa từ, hoặc xây dựng hệ thống thông minh hơn.")

# Bài 2 Bình luận trận đấu

print("\n===== BÀI 2 =====")

# - Nạp dữ liệu, kiểm tra missing values.
matches = pd.read_csv(r'd:\FPT\ITA105\LMS-ITA105\Lab4\Lab4\ITA105_Lab_4_Match_comments.csv', encoding='utf-8')
print(matches.shape)
print(matches.isnull().sum())
print(matches.head())

# - Tiền xử lý văn bản bình luận: lowercase, tokenization, loại bỏ stop words.

matches['cleaned_comment'] = matches['comment_text'].apply(clean_text)
print(matches[['comment_text', 'cleaned_comment']].head())

# - Tạo TF-IDF matrix từ bình luận.
matches['comment_text'] = matches['comment_text'].fillna("")
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(matches['comment_text'])
print("\nTF-IDF matrix:")
print(tfidf_matrix.toarray())

# - Tạo Word2Vec embeddings và tìm từ gần nghĩa với “xuất sắc”.
model = Word2Vec(sentences=[text.split() for text in matches['cleaned_comment']], vector_size=50, window=2, min_count=1)
similar_words = model.wv.most_similar("xuất", topn=5)
print("\n5 words similar to 'xuất':", similar_words)

# - So sánh TF-IDF vs Word2Vec về khả năng biểu diễn ý nghĩa của bình luận.
print("\nKết luận:")
print("TF-IDF: tốt cho các mô hình truyền thống, khi mục tiêu là phân loại hoặc tìm kiếm dựa trên từ khóa.")
print("Word2Vec: mạnh hơn khi bạn muốn hiểu ngữ nghĩa, quan hệ giữa từ, hoặc xây dựng hệ thống thông minh (ví dụ: gợi ý từ đồng nghĩa, phân tích cảm xúc sâu hơn).")

# Bài 3 Feedback người chơi

print("\n===== BÀI 3 =====")

# - Nạp dữ liệu, kiểm tra missing values.
feedback = pd.read_csv(r'd:\FPT\ITA105\LMS-ITA105\Lab4\Lab4\ITA105_Lab_4_Player_feedback.csv', encoding='utf-8')
print(feedback.shape)
print(feedback.isnull().sum())
print(feedback.head())

# - Label / One-Hot Encoding cho các thuộc tính categorical.
categorical_cols = [col for col in ['game_name', 'player_type'] if col in feedback.columns]
feedback_encoded = pd.get_dummies(feedback, columns=categorical_cols, drop_first=True)
print(feedback_encoded.head())

# - Tiền xử lý văn bản: lowercase, tokenization, loại bỏ stop words.
feedback_encoded['cleaned_feedback'] = feedback_encoded['feedback_text'].apply(clean_text)
print(feedback_encoded[['feedback_text', 'cleaned_feedback']].head())

# - Tạo TF-IDF matrix cho feedback.

feedback_encoded['feedback_text'] = feedback_encoded['feedback_text'].fillna("")
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(feedback_encoded['feedback_text'])
print("\nTF-IDF matrix:")
print(tfidf_matrix.toarray())

# - Tạo Word2Vec embeddings và tìm từ gần nghĩa với “đẹp”.
model = Word2Vec(sentences=[text.split() for text in feedback_encoded['cleaned_feedback']], vector_size=50, window=2, min_count=1)
similar_words = model.wv.most_similar("đẹp", topn=5)
print("\n5 words similar to 'đẹp':", similar_words)

# - Lựa chọn TF-IDF hay Word2Vec để phân loại sentiment người chơi, vì sao?
print("\nKết luận:")
print("Nếu mục tiêu là phân loại sentiment cơ bản (tích cực/tiêu cực) với dữ liệu vừa phải → TF-IDF là lựa chọn hợp lý.")
print("Nếu muốn hiểu ngữ nghĩa sâu hơn, phát hiện sắc thái cảm xúc đa dạng, hoặc dữ liệu lớn → Word2Vec sẽ hiệu quả hơn.")

# Bài 4 Review album

print("\n===== BÀI 4 =====")

# - Nạp dữ liệu, kiểm tra missing values.
albums = pd.read_csv(r'd:\FPT\ITA105\LMS-ITA105\Lab4\Lab4\ITA105_Lab_4_Album_reviews.csv', encoding='utf-8')
print(albums.shape)
print(albums.isnull().sum())
print(albums.head())

# - Label / One-Hot Encoding các cột categorical.
categorical_cols = [col for col in ['album_name', 'artist_name'] if col in albums.columns]
albums_encoded = pd.get_dummies(albums, columns=categorical_cols, drop_first=True)  
print(albums_encoded.head())

# - Tiền xử lý review: lowercase, tokenization, loại bỏ stop words.
albums_encoded['cleaned_review'] = albums_encoded['review_text'].apply(clean_text)
print(albums_encoded[['review_text', 'cleaned_review']].head())

# - Tạo TF-IDF matrix cho review.
albums_encoded['review_text'] = albums_encoded['review_text'].fillna("")
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(albums_encoded['review_text'])
print("\nTF-IDF matrix:")
print(tfidf_matrix.toarray())

# - Tạo Word2Vec embeddings và tìm từ gần nghĩa với “sáng tạo”.
model = Word2Vec(sentences=[text.split() for text in albums_encoded['cleaned_review']], vector_size=50, window=2, min_count=1)
similar_words = model.wv.most_similar("sáng", topn=5)
print("\n5 words similar to 'sáng':", similar_words)
