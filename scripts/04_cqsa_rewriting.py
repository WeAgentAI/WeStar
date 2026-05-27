"""
WeStar Step 4: CQSA Stylized Answer Rewriting
Rewrite standard CQA answers into stylized versions matching target cluster's style.
"""
import json
import os
import argparse
import yaml
from tqdm import tqdm
from utils import call_llm, load_jsonl, save_jsonl


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_style_examples(biz_id, comments_dir, max_examples=5):
    """Get style reference examples for a given account."""
    comments_path = os.path.join(comments_dir, f"{biz_id}.jsonl")
    if not os.path.exists(comments_path):
        return ""
    
    comments = load_jsonl(comments_path)[:max_examples]
    examples = []
    for c in comments:
        examples.append(f"问题：{c['question']}\n回答：{c['answer']}")
    return "\n\n".join(examples)


def rewrite_answer(cqa_item, style_labels, style_examples, prompt_template):
    """Rewrite a CQA answer to match target style."""
    prompt = prompt_template.replace("{{twelve_labels}}", json.dumps(style_labels, ensure_ascii=False))
    prompt = prompt.replace("{{examples}}", style_examples)
    prompt = prompt.replace("{{content}}", cqa_item.get('context', ''))
    prompt = prompt.replace("{{question}}", cqa_item['question'])
    prompt = prompt.replace("{{answer}}", cqa_item['answer'])
    
    response = call_llm(prompt)
    return response


def main():
    parser = argparse.ArgumentParser(description="WeStar CQSA Rewriting")
    parser.add_argument("--config", type=str, default="configs/rewrite_config.yaml")
    parser.add_argument("--cluster_id", type=str, default=None, help="Process specific cluster")
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    # Load data
    cqa_path = config['data']['cqa_path']
    styles_path = config['data']['styles_path']
    label_path = config['data']['label_path']
    comments_dir = config['data']['comments_dir']
    output_dir = config['output']['output_dir']
    
    cqa_data = load_jsonl(cqa_path)
    styles_data = {item['id']: item['style'] for item in load_jsonl(styles_path)}
    
    with open(label_path, 'r') as f:
        label_map = json.load(f)
    
    # Load prompt template
    with open(config['prompts']['cqsa_rewriting'], 'r') as f:
        rewrite_prompt = f.read()
    
    # Group CQA data by cluster
    cluster_cqa = {}
    for item in cqa_data:
        biz_id = item.get('biz_id', '')
        if biz_id not in label_map:
            continue
        cluster_id = str(label_map[biz_id])
        if args.cluster_id and cluster_id != args.cluster_id:
            continue
        if cluster_id not in cluster_cqa:
            cluster_cqa[cluster_id] = []
        cluster_cqa[cluster_id].append(item)
    
    # Process each cluster
    for cluster_id, items in cluster_cqa.items():
        print(f"\n=== Processing Cluster {cluster_id} ({len(items)} items) ===")
        
        cluster_output_dir = os.path.join(output_dir, cluster_id)
        os.makedirs(cluster_output_dir, exist_ok=True)
        
        results = []
        for item in tqdm(items, desc=f"Cluster {cluster_id}"):
            biz_id = item['biz_id']
            style_labels = styles_data.get(biz_id, {})
            style_examples = get_style_examples(biz_id, comments_dir)
            
            rewritten = rewrite_answer(item, style_labels, style_examples, rewrite_prompt)
            
            item['rewrite_answer'] = rewritten
            results.append(item)
        
        # Save
        output_path = os.path.join(cluster_output_dir, "cqsa_data.jsonl")
        save_jsonl(results, output_path)
        print(f"  Saved {len(results)} CQSA instances → {output_path}")


if __name__ == "__main__":
    main()
