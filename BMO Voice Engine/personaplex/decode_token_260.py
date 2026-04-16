import sentencepiece as spm
sp = spm.SentencePieceProcessor(model_file="tokenizer_spm_32k_3.model")
print({0: (sp.id_to_piece(0), sp.decode_ids([0])), 260: (sp.id_to_piece(260), sp.decode_ids([260]))})