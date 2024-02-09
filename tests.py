import sentencepiece as spm

src_sp = spm.SentencePieceProcessor()

print(src_sp.encode(['This is a test', 'Hello world'], out_type=int))