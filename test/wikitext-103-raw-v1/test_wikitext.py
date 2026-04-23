import os
import sys
import argparse

import torch
import numpy as np

sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")))
from model import RWKV_RNN


def run_inference_and_save(
        model_path,
        input_tokens_npy,
        input_logits_npy,
        output_logits_npy,
        output_tokens_npy,
        device='cuda',
        batch_size=1,
        max_data_items=8,
        length_token_input=512,
        length_token_output=512,
        top_k=512
    ):
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

    # 控制读入的数据项个数（用于本地调试）
    MAX_DATA_ITEMS = max_data_items  # 设置为 None 表示读取全部数据

    # 控制输出的 token 长度
    LENGTH_TOKEN_INPUT = length_token_input # 设置输入 token 长度参数
    LENGTH_TOKEN_OUTPUT = length_token_output  # 设置输出长度参数

    # 控制保存 logits 的 topk
    TOP_K = top_k  # 设置保存的 topk 值

    # 加载 tokens 数据，限制最大长度
    input_tokens = np.load(input_tokens_npy, allow_pickle=True)[:MAX_DATA_ITEMS] if MAX_DATA_ITEMS else np.load(input_tokens_npy, allow_pickle=True)

    # 断言 LENGTH_TOKEN_INPUT 小于等于 input_tokens 中最短数据的长度
    min_length = min(len(tokens) for tokens in input_tokens)
    assert LENGTH_TOKEN_INPUT <= min_length, "LENGTH_TOKEN_INPUT 必须小于等于输入数据集中最短序列的长度"

    # 将输入数据转换为定长数组
    input_tokens = torch.from_numpy(np.array([tokens[:LENGTH_TOKEN_INPUT] for tokens in input_tokens], dtype=np.int32)).to(device)

    # 初始化保存 logits 和 tokens 的数组为 torch.tensor
    input_logits = torch.zeros((MAX_DATA_ITEMS, LENGTH_TOKEN_INPUT, TOP_K), device=device, dtype=torch.float32)
    output_tokens = torch.zeros((MAX_DATA_ITEMS, LENGTH_TOKEN_OUTPUT), device=device, dtype=torch.int32)
    output_logits = torch.zeros((MAX_DATA_ITEMS, LENGTH_TOKEN_OUTPUT, TOP_K), device=device, dtype=torch.float32)

    # 检查是否存在中间结果文件
    if os.path.exists(input_logits_npy):
        loaded_input_logits = torch.tensor(np.load(input_logits_npy, allow_pickle=True), device=device, dtype=torch.float32)
        input_logits[:loaded_input_logits.shape[0]] = loaded_input_logits
        if (start_idx := loaded_input_logits.shape[0]) >= MAX_DATA_ITEMS:
            print("检测到结果文件且已处理完所有数据，退出推理")
            return
        print(f"检测到中间结果文件，从第 {start_idx} 条继续推理...")
    else:
        start_idx = 0

    # 加载模型
    model = RWKV_RNN(args).to(device)
    model.eval()

    num_batches = len(input_tokens) // batch_size + (1 if len(input_tokens) % batch_size != 0 else 0)  # 计算批次数量

    for batch_idx in range(start_idx // batch_size, num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(input_tokens))
        tokens = input_tokens[start:end]

        # 推理：逐 token 输入，获得 input_logits
        for token_idx in range(LENGTH_TOKEN_INPUT):
            with torch.no_grad():
                logits = model.forward(tokens[:, token_idx])
                topk_values, _ = torch.topk(logits, TOP_K, dim=-1)
                input_logits[start:end, token_idx] = topk_values

        # 循环预测 LENGTH_TOKEN_OUTPUT 个 token
        for output_idx in range(LENGTH_TOKEN_OUTPUT):
            with torch.no_grad():
                logits = model.forward(tokens[:, -1])
                topk_values, _ = torch.topk(logits, TOP_K, dim=-1)
                output_logits[start:end, output_idx] = topk_values
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.argmax(probs, dim=-1)
                tokens = torch.cat([tokens, next_token.unsqueeze(-1)], dim=1)
                output_tokens[start:end, output_idx] = next_token

        # 保存中间结果
        np.save(input_logits_npy, input_logits[:end].cpu().numpy())
        np.save(output_tokens_npy, output_tokens[:end].cpu().numpy())
        np.save(output_logits_npy, output_logits[:end].cpu().numpy())

        # 打印进度
        print(f"已处理 {end} / {len(input_tokens)} 条数据")

    # 保存最终结果
    np.save(input_logits_npy, input_logits.cpu().numpy())
    np.save(output_tokens_npy, output_tokens.cpu().numpy())
    np.save(output_logits_npy, output_logits.cpu().numpy())
    print(f"全部完成，输出已保存到 {input_logits_npy}, {output_tokens_npy}, 和 {output_logits_npy}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--delete-outputs', action='store_true', help='删除目录下所有输出文件')
    args = parser.parse_args()

    # 推理参数
    model_path = "E:\\Resources\\Models\\rwkv-7\\RWKV-x070-World-0.1B-v2.8-20241210-ctx4096"  # 替换为实际模型路径
    input_tokens_npy = "test\\wikitext-103-raw-v1\\sampled_1000_tokens.npy"
    input_logits_npy = "test\\wikitext-103-raw-v1\\input_logits.npy"
    output_tokens_npy = "test\\wikitext-103-raw-v1\\output_tokens.npy"
    output_logits_npy = output_tokens_npy.replace("output_tokens", "output_logits")
    batch_size = 4

    # 如果设置了删除输出文件，清理目录
    if args.delete_outputs:
        for file_path in [input_logits_npy, output_tokens_npy, output_logits_npy]:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"已删除文件: {file_path}")

    with torch.no_grad():
        run_inference_and_save(model_path, input_tokens_npy, input_logits_npy, output_logits_npy, output_tokens_npy, batch_size=batch_size)
