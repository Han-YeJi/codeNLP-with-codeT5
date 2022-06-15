import torch
import numpy as np


def create_encoding(df,tokenizer):
    test_encoded_dict = tokenizer(
        df['code1'].tolist(),
        df['code2'].tolist(),

        max_length = 512,
        truncation = True,
        return_token_type_ids = True,
        padding = "max_length",
        return_tensors = 'pt',
      )
    
    return test_encoded_dict


