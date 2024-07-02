import torch
from transformers import BertTokenizer, BertForSequenceClassification
import argparse
import os
import numpy as np
from matplotlib import pyplot as plt
import re
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载预训练模型
model_path = 'best_model.pth'
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=6)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# 定义情绪标签映射
label_map = {0: 'neutral', 1: 'happy', 2: 'angry', 3: 'sad', 4: 'fear', 5: 'surprise'}


def predict(text, top_k=1, with_scores=False):
    """
    预测输入文本的情绪类别，并返回前 top_k 个预测结果及其对应的分数（如果需要）。

    参数:
    - text (str): 输入的文本字符串。
    - top_k (int): 要返回的前 k 个预测结果。默认为 1。
    - with_scores (bool): 是否返回预测结果的分数。默认为 False。

    返回:
    - list: 前 top_k 个预测结果，如果 with_scores 为 True，则返回 (label, score) 元组列表，否则返回 label 列表。
    """
    # 对输入文本进行编码
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # 禁用梯度计算
    with torch.no_grad():
        # 模型前向传播，获取logits
        outputs = model(**inputs)
        logits = outputs.logits

        # 计算每个类别的概率分数
        probs = F.softmax(logits, dim=-1)

        # 获取概率最大的前 top_k 个类别及其分数
        top_probs, top_indices = torch.topk(probs, top_k)

        results = []
        for i in range(top_k):
            label = label_map[top_indices[0][i].item()]
            score = top_probs[0][i].item()
            if with_scores:
                results.append((label, score))
            else:
                results.append(label)
    return results


def predict_from_file(file_path, output_file, top_k=1, with_scores=False):
    """
    从文件中读取文本并预测每行文本的情绪类别，输出前 top_k 个预测结果及其对应的分数（如果需要）。

    参数:
    - file_path (str): 输入的文本文件路径。
    - output_file (str): 输出结果的文件路径。
    - top_k (int): 要返回的前 k 个预测结果。默认为 1。
    - with_scores (bool): 是否返回预测结果的分数。默认为 False。

    返回:
    - None
    """
    # 读取输入文件中的所有文本
    with open(file_path, 'r', encoding='utf-8') as file:
        texts = file.readlines()
        sentence_list = []
        sentence_enders = re.compile('[。!?；;\n\r]+|[.?!]+')
        for i in range(len(texts)):
            texts[i] = texts[i].strip()
            if texts[i]:
                for sent in sentence_enders.split(texts[i]):
                    if sent:
                        sentence_list.append(sent)
        print(sentence_list)

    results = []
    all_predictions = []
    for text in sentence_list:
        if text:  # 忽略空行
            # 预测每行文本的情绪
            predictions = predict(text, top_k=6, with_scores=with_scores)
            all_predictions.append(predictions)  # 记录所有预测结果,用于绘制趋势曲线
            if with_scores:
                prediction_str = ", ".join([f"{label}: {score:.4f}" for label, score in predictions[:top_k]])
            else:
                prediction_str = ", ".join(predictions)
            results.append(f"Text: {text} -> Predictions: {prediction_str}")

    # 将预测结果写入输出文件
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write("\n".join(results))
    print(f"Results written to {output_file}")
    return all_predictions

# 绘制六种情感的趋势曲线
def plot_results(emotion_scores: list):
    num_sentences = len(emotion_scores)

    # 初始化每种情感的分数列表
    emotion_trends = {emotion: [] for emotion in label_map.values()}

    # 填充情感分数
    for scores_items in emotion_scores:
        for emotion_score in scores_items:
            emotion_trends[emotion_score[0]].append(emotion_score[1])

    # 绘制折线图
    plt.figure(figsize=(10, 6))
    for emotion, scores in emotion_trends.items():
        plt.plot(range(num_sentences), scores, label=emotion)

    plt.xlabel('Sentence Number')
    plt.ylabel('Emotion Score')
    plt.title('Emotion Trends')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':

    if __name__ == '__main__':
        parser = argparse.ArgumentParser(description="Emotion prediction script")
        parser.add_argument('--text', type=str, help='直接输入文本进行情绪预测')
        parser.add_argument('--file', type=str, help='输入一个文本文件进行情绪预测')
        parser.add_argument('--output', type=str, default='output.txt',
                            help='保存预测结果的文件路径，默认为 output.txt')
        parser.add_argument('--top_k', type=int, default=1, help='前 k 个预测结果')
        parser.add_argument('--with_scores', action='store_true', help='是否返回预测结果的分数')
        parser.add_argument('--plot', action='store_true', help='是否绘制情感趋势图')
        args = parser.parse_args()

        if args.text:
            predictions = predict(args.text, top_k=args.top_k, with_scores=args.with_scores)
            if args.with_scores:
                prediction_str = ", ".join([f"{label}: {score:.4f}" for label, score in predictions])
            else:
                prediction_str = ", ".join(predictions)
            print(f"Text: {args.text} -> Predictions: {prediction_str}")
        elif args.file:
            if not os.path.exists(args.file):
                print(f"File {args.file} not found.")
            else:
                all_predictions = predict_from_file(args.file, args.output, top_k=args.top_k,
                                                    with_scores=args.with_scores)
                if args.plot:
                    plot_results(all_predictions)
        else:
            print("Please provide either --text or --file argument.")

    # # 非终端测试
    # r = predict_from_file("2024年如何重新评价《原神》这款游戏？.txt", "result.txt", top_k=3, with_scores=True)
    # plot_results(r)
    # predict("2024年如何重新评价《原神》这款游戏？", top_k=3, with_scores=True)
# 示例
# python test.py --file .txt --output result.txt --top_k 3 --with_scores
# python test.py --text "这是一个测试文本" --top_k 3 --with_scores