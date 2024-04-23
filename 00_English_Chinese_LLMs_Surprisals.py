import pandas as pd
import numpy as np
import re
import math
import os
import string
import shutil, sys, glob

data = ('/your data folder/')

!pip install minicons

from minicons import cwe, scorer
import torch
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login

# your huggingface login
login("HERE")

available_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
gpu_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(available_gpus)
print(gpu_names)

gpt2Chinese = scorer.IncrementalLMScorer('uer/gpt2-chinese-cluecorpussmall', 'cuda')
bertChinese = scorer.IncrementalLMScorer('google-bert/bert-base-chinese', 'cuda')
llamaChinese = scorer.IncrementalLMScorer('hfl/chinese-llama-2-7b', 'cuda') 
gpt2English = scorer.IncrementalLMScorer('gpt2', 'cuda')
bertEnglish = scorer.IncrementalLMScorer('google-bert/bert-base-uncased')
llamaEnglish = scorer.IncrementalLMScorer('meta-llama/Llama-2-7b-hf',"cuda")

"""# utility functions"""

def response_surp_mean(response, len, model):
  if len > 1:  
    score = model.sequence_score([response], reduction = lambda x: -x.mean(0).item())
    return round(score[0],2)
  else:
    return np.nan

"""# Chinese and English dataset surprisals"""

chinese = pd.read_csv(data + 'Chinese.csv', index_col = 0)
english = pd.read_csv(data + 'English.csv', index_col = 0)

chinese['gpt2_surp_mean'] = chinese.apply(lambda x: response_surp_mean(x.content_clean, x.clean_utterance_len, gpt2Chinese), axis=1)
chinese['llama_surp_mean'] = chinese.apply(lambda x: response_surp_mean(x.content_clean, x.clean_utterance_len, llamaChinese), axis=1)
chinese['bert_surp_mean'] = chinese.apply(lambda x: response_surp_mean(x.content_clean, x.clean_utterance_len, bertChinese), axis=1)
english['gpt2_surp_mean'] = english.apply(lambda x: response_surp_mean(x.content_clean, x.clean_utterance_len, gpt2English), axis=1)
english['llama_surp_mean'] = english.apply(lambda x: response_surp_mean(x.content_clean, x.clean_utterance_len, llamaEnglish), axis=1)
english['bert_surp_mean'] = english.apply(lambda x: response_surp_mean(x.content_clean, x.clean_utterance_len, bertEnglish), axis=1)

chinese.to_csv(data + 'Chinese.csv')
english.to_csv(data + 'English.csv')





