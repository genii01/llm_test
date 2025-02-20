import sentencepiece as spm

# 1. 데이터 파일 생성
with open("sample.txt", "w") as f:
    f.write("This is a simple example of SentencePiece tokenizer.\n")
    f.write("SentencePiece allows us to train a tokenizer from raw text.\n")

# 2. SentencePiece 모델 학습
spm.SentencePieceTrainer.train(
    input="sample.txt", model_prefix="bpe", vocab_size=50, model_type="bpe"
)

# 3. 학습된 모델 로드
sp = spm.SentencePieceProcessor()
sp.load("bpe.model")

# 4. 토큰화 예제
text = "This is a simple example of SentencePiece tokenizer."
tokens = sp.encode_as_pieces(text)
ids = sp.encode_as_ids(text)

print("Original Text:", text)
print("Tokenized:", tokens)
print("Token IDs:", ids)

# 5. 디코딩 (원본 문장 복원)
decoded_text = sp.decode_pieces(tokens)
print("Decoded:", decoded_text)
