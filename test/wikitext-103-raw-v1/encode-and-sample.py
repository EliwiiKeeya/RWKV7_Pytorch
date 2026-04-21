# -*- encoding: utf-8 -*-
# @File			: encode-and-sample.py
# @Date			: 2026/04/21 20:46:33
# @Author		: Eliwii_Keeya

"""
编码训练集中的文本数据，保存输出字符串和输出向量。
1. 读取之前保存的 WikiText 数据集文件。 
2. 使用 RWKV_TOKENIZER 对文本进行编码，得到 token 列表。
3. 将编码后的 token 列表保存到 JSONL 和 NPY 文件中。
4. 从编码后的数据中筛选出满足条件的样本（如 token 数量超过一定阈值），随机抽取指定数量的样本。
5. 将随机抽取的样本的 token 列表保存到 JSONL 和 NPY 文件中。
"""

import os
import sys
import json
import random
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")))
from tokenizer import RWKV_TOKENIZER

# 路径配置
TRAIN_PATHS = [
    'test/wikitext-103-raw-v1/data/train-00000-of-00002-b755d19de94348c6.parquet',
    'test/wikitext-103-raw-v1/data/train-00001-of-00002-0bf6d0c487c2e75b.parquet',
]
SAMPLE_NUM = 1000
TOKEN_LENGTH_THRESHOLD = 512
SAMPLED_TOKENS_JSONL = f'test/wikitext-103-raw-v1/sampled_{SAMPLE_NUM}_tokens.jsonl'
SAMPLED_TOKENS_NPY = f'test/wikitext-103-raw-v1/sampled_{SAMPLE_NUM}_tokens.npy'
ALL_TOKENS_JSONL = f'test/wikitext-103-raw-v1/all_encoded_tokens.jsonl'
ALL_TOKENS_NPY = f'test/wikitext-103-raw-v1/all_encoded_tokens.npy'

# 初始化分词器
tokenizer = RWKV_TOKENIZER()

# 读取所有训练集 parquet 文件
dfs = []
total = 0
for path in TRAIN_PATHS:
    print(f'加载 {path} ...')
    df = pd.read_parquet(path)
    print(f'  样本数: {len(df)}')
    dfs.append(df)
    total += len(df)

df_all = pd.concat(dfs, ignore_index=True)
print(f'训练集总样本数: {len(df_all)}')

# 分词函数
def encode_text(row):
    text = row.get('text', '')
    tokens = tokenizer.encode(text)
    return {"text": text, "tokens": tokens}


# 主函数
if __name__ == "__main__":
    # 使用多进程进行分词
    print("开始分词...")
    with Pool(cpu_count()) as pool:
        results = list(tqdm(pool.imap(
            encode_text, [row for _, row in df_all.iterrows()]), total=len(df_all)))

    # 所有样本保存为 jsonl
    print("保存所有的 encoded token 到 JSONL 文件...")
    with open(ALL_TOKENS_JSONL, 'w', encoding='utf-8') as f:
        for item in results:
            json.dump(
                {"text": item["text"], "tokens": item["tokens"]}, f, ensure_ascii=False)
            f.write('\n')

    # 所有样本保存为 npy
    print("保存所有的 encoded token 到 NPY 文件...")
    all_tokens = [item["tokens"] for item in results]
    np.save(ALL_TOKENS_NPY, np.array(all_tokens, dtype=object))
    print(f'所有 encoded token 已保存到 {ALL_TOKENS_JSONL} 和 {ALL_TOKENS_NPY}')

    # 筛选满足条件的数据
    print("筛选满足条件的数据...")
    filtered_data = [item for item in results if len(
        item["tokens"]) >= TOKEN_LENGTH_THRESHOLD]

    if len(filtered_data) < SAMPLE_NUM:
        raise ValueError(
            f'满足条件的数据量不足 {SAMPLE_NUM} 条！实际数量: {len(filtered_data)}')
    
    # 随机抽取 SAMPLE_NUM 条数据
    print("随机抽取样本...")
    sampled = random.sample(filtered_data, SAMPLE_NUM)

    # 随机样本保存为 jsonl
    print("保存随机抽取样本的 token 到 JSONL 文件...")
    with open(SAMPLED_TOKENS_JSONL, 'w', encoding='utf-8') as f:
        for item in sampled:
            json.dump(
                {"text": item["text"], "tokens": item["tokens"]}, f, ensure_ascii=False)
            f.write('\n')

    # 随机样本保存为 npy
    print("保存随机抽取样本的 token 到 NPY 文件...")
    sampled_tokens = [item["tokens"] for item in sampled]
    np.save(SAMPLED_TOKENS_NPY, np.array(sampled_tokens, dtype=object))
    print(f'随机抽取样本的 token 已保存到 {SAMPLED_TOKENS_JSONL} 和 {SAMPLED_TOKENS_NPY}')
