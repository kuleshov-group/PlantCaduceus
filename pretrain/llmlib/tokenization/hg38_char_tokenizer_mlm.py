""" 
From: https://github.com/dariush-bahrami/character-tokenizer/blob/master/charactertokenizer/core.py

CharacterTokenzier for Hugging Face Transformers.
This is heavily inspired from CanineTokenizer in transformers package.
"""
import json
import os
import sys
from transformers import BertTokenizerFast
from transformers.tokenization_utils import AddedToken
    

if __name__ == '__main__':
    model_max_length = None if len(sys.argv) < 2 else int(sys.argv[1])# sys.argv[1]
    characters = ['A', 'C', 'G', 'T', 'N']
    bos_token = AddedToken("[BOS]", lstrip=False, rstrip=False)
    eos_token = AddedToken("[SEP]", lstrip=False, rstrip=False)
    sep_token = AddedToken("[SEP]", lstrip=False, rstrip=False)
    cls_token = AddedToken("[CLS]", lstrip=False, rstrip=False)
    pad_token = AddedToken("[PAD]", lstrip=False, rstrip=False)
    unk_token = AddedToken("[UNK]", lstrip=False, rstrip=False)

    mask_token = AddedToken("[MASK]", lstrip=True, rstrip=False)

    out_path = f"llmlib/tokenization/autotoks/hg38_char_tokenizer_maxlen_{model_max_length}/"
    os.makedirs(out_path)

    vocab = {
        "[PAD]": 0,
        "[UNK]": 1,
        "[CLS]": 2,
        "[SEP]": 3,
        "[MASK]": 4,
        "[RESERVED]": 5,
        "[BOS]": 6,
        "A": 7,
        "C": 8,
        "G": 9,
        "T": 10,
        "N": 11
        # Add any additional characters or tokens here
    }
    vocab_file_path = out_path + '/tmp_vocab.txt'
    with open(vocab_file_path, 'w') as f:
        f.write("\n".join(vocab.keys()))
    
    padding_side: str='left'
    
    tokenizer = BertTokenizerFast(
            vocab_file=vocab_file_path,
            bos_token=bos_token,
            eos_token=sep_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            unk_token=unk_token,
            add_prefix_space=False,
            model_max_length=model_max_length,
            padding_side=padding_side,
        )
    #os.remove(vocab_file_path)
    tokenizer.save_pretrained(out_path)
    tokenizer.from_pretrained(out_path)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(out_path)
    print(tokenizer.get_vocab())
    assert tokenizer.get_vocab() == vocab
    tokenizer.encode("atcg")
    tokenizer.encode("[UNK]")
    tokenizer.encode("test")
