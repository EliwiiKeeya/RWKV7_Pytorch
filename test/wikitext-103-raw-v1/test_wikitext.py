import os
import sys
import json

import torch
import numpy as np

sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")))
from model import RWKV_RNN


def run_inference_and_save(model_path, input_npy, output_logits_npy, device='cuda', batch_size=1):
    """
    读取 input_npy，批量做推理，保存输出 logits 和输出向量。
    支持从中断处恢复。
    输出 logits 格式调整为 [input_index, input_token, vocab_size]。
    """
    args = {
        'MODEL_NAME': model_path,
        'vocab_size': 65536,
        'batch_size': batch_size,
        'device': device
    }

    # 常量：控制读入的数据项个数（用于本地调试）
    MAX_DATA_ITEMS = 8  # 设置为 None 表示读取全部数据

    # 加载 tokens 数据，限制最大长度
    LENGTH_TOKEN_INPUT = 512  # 设置最大长度参数
    all_tokens = np.load(input_npy, allow_pickle=True)[:MAX_DATA_ITEMS] if MAX_DATA_ITEMS else np.load(input_npy, allow_pickle=True)

    # 断言输入数据长度大于 LENGTH_TOKEN_INPUT
    assert all(len(tokens) >= LENGTH_TOKEN_INPUT for tokens in all_tokens), "输入数据中存在长度小于 LENGTH_TOKEN_INPUT 的序列。"

    # 将输入数据组织为定长数组
    all_tokens = np.array([tokens[:LENGTH_TOKEN_INPUT] for tokens in all_tokens])

    # 初始化 saved_logits 为全零数组
    saved_logits = np.zeros((MAX_DATA_ITEMS, LENGTH_TOKEN_INPUT, args['vocab_size']), dtype=np.float32)

    # 检查是否存在中间结果文件
    if os.path.exists(output_logits_npy):
        loaded_logits = np.load(output_logits_npy, allow_pickle=True)
        saved_logits[:loaded_logits.shape[0]] = loaded_logits
        if (start_idx:= loaded_logits.shape[0]) >= MAX_DATA_ITEMS:
            print(f"检测到结果文件且已处理完所有数据，退出推理")
            return
        print(f"检测到中间结果文件，从第 {start_idx} 条继续推理...")
    else:
        start_idx = 0

    # 加载模型
    model = RWKV_RNN(args).to(device)
    model.eval()

    all_vectors = []  # 用于存储所有输出向量
    num_batches = len(all_tokens) // batch_size + (1 if len(all_tokens) % batch_size != 0 else 0)  # 计算批次数量

    for batch_idx in range(start_idx // batch_size, num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(all_tokens))  # 处理最后一批数据
        tokens = torch.tensor(all_tokens[start:end], device=device)

        # 推理：逐 token 输入，获得最后输出
        for token_idx in range(LENGTH_TOKEN_INPUT):
            with torch.no_grad():
                out = model.forward(tokens[:, token_idx])  # 输入 [batch_size]，输出 [batch_size, vocab_size]
                saved_logits[start:end, token_idx, :] = out.detach().cpu().numpy()

        # logits 落盘
        np.save(output_logits_npy, saved_logits[:end, :, :])

        # 保存输出向量
        for token_seq_logits in saved_logits[start:end]:
            all_vectors.append(token_seq_logits[-1])  # 假设最后一个 logits 作为输出向量

        print(f'已处理 {end} / {len(all_tokens)} 条')

    # 保存 logits 为 npy 文件
    np.save(output_logits_npy, saved_logits)
    print(f'全部完成，输出已保存到 {output_logits_npy}')


if __name__ == '__main__':
    # 推理参数
    model_path = "E:\\Resources\\Models\\rwkv-7\\RWKV-x070-World-0.1B-v2.8-20241210-ctx4096"  # 替换为实际模型路径
    input_npy = "test\\wikitext-103-raw-v1\\sampled_1000_tokens.npy"
    output_logits_npy = "test\\wikitext-103-raw-v1\\inference_logits.npy"
    batch_size = 8

    with torch.no_grad():
        run_inference_and_save(model_path, input_npy, output_logits_npy, batch_size=batch_size)
