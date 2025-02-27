import os
import re
import json
import csv
import torch
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer
from model import BRNNAttClassifcationModelV2

def parse_args_from_log(log_file):
    with open(log_file, 'r') as f:
        first_line = f.readline()
    match = re.search(r'Namespace\((.*?)\)', first_line)
    if match:
        args_dict = {}
        args_str = match.group(1)
        for item in args_str.split(', '):
            key, value = item.split('=', 1)
            key = key.strip()
            value = value.strip().strip("'")
            if value.isdigit():
                value = int(value)
            elif value.replace('.', '', 1).isdigit():
                value = float(value)
            args_dict[key] = value
        return type('Args', (object,), args_dict)()
    else:
        raise ValueError("Could not parse args from log file")

def get_latest_checkpoint(save_path):
    bin_files = [f for f in os.listdir(save_path) if f.endswith(".bin")]
    if not bin_files:
        raise FileNotFoundError("No .bin checkpoint files found in save_path")
    latest_file = sorted(bin_files, key=lambda x: os.path.getctime(os.path.join(save_path, x)))[-1]
    return os.path.join(save_path, latest_file)

def encode_text_input(tokenizer, args, text: list):
    encoded_inputs = tokenizer(text, max_length=args.bert_max_length, padding=args.bert_padding, truncation=True)
    
    input_ids = torch.tensor(encoded_inputs['input_ids'], dtype=torch.int64, device=args.device)
    attention_mask = torch.tensor(encoded_inputs['attention_mask'], dtype=torch.int64, device=args.device)
    token_type_ids = torch.tensor(encoded_inputs['token_type_ids'], dtype=torch.int64, device=args.device)

    data = dict(
        input_ids = input_ids, 
        attention_mask = attention_mask,
        token_type_ids = token_type_ids
    )
    return data

def inference(model, tokenizer, args, text: list):
    inputs = encode_text_input(tokenizer, args, text)
    with torch.no_grad():
        prediction = model(inputs, inference=True)
    return prediction.cpu().numpy().tolist()

if __name__ == "__main__":
    save_path = 'save/qunaer_20250226_balance82/models--nghuyong--ernie-3.0-base-zh/feature2_fusionweighted_advnone_20250227112838'
    val_file = 'data/qunaer_20252026_all.csv'

    train_log = os.path.join(save_path, 'train.log')
    checkpoint_file = get_latest_checkpoint(save_path)
    args = parse_args_from_log(train_log)

    tokenizer = AutoTokenizer.from_pretrained(args.bert_dir)
    model = BRNNAttClassifcationModelV2(args)

    checkpoint = torch.load(checkpoint_file, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(args.device)
    model.eval()

    val_file_name = os.path.splitext(os.path.basename(val_file))[0]
    output_file = os.path.join(os.path.dirname(val_file), f"{val_file_name}_pred.csv")

    inference_data = []

    if val_file.endswith('.json'):
        inference_data = json.load(open(val_file, 'r', encoding='utf-8'))
    elif val_file.endswith('.csv'):
        with open(val_file, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                inference_data.append(row)
    else:
        inference_data = [{'reviewtext': "这家酒店太差了，不推荐"}, {'reviewtext': "天空是蓝色的，云朵是白色的"}, {'reviewtext': "这家酒店很好，推荐"}]
    
    for item in tqdm(inference_data):
        text = item['详细评价']
        prediction = inference(model, tokenizer, args, [text])
        item['prediction'] = prediction[0]

    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=inference_data[0].keys())
        writer.writeheader()
        writer.writerows(inference_data)