"""
WeStar Step 5: Quality Scoring & Filtering
Score each CQSA instance on 4 dimensions and select top-quality samples.
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


def score_cqsa(item, style_labels, style_examples, scoring_prompt):
    """Score a CQSA instance on 4 dimensions using LLM-as-a-judge."""
    prompt = scoring_prompt.replace("{{content}}", item.get('context', ''))
    prompt = prompt.replace("{{style_labels}}", json.dumps(style_labels, ensure_ascii=False))
    prompt = prompt.replace("{{style_examples}}", style_examples)
    prompt = prompt.replace("{{question}}", item['question'])
    prompt = prompt.replace("{{rewrite_answer}}", item.get('rewrite_answer', ''))
    
    response = call_llm(prompt)
    
    try:
        # Parse scores from response (handles </think> tags from DeepSeek-R1)
        if '</think>' in response:
            response = response.split('</think>')[1].strip()
        scores_str = response.split('```json')[1].split('```')[0].strip()
        scores = json.loads(scores_str)
        return scores
    except (json.JSONDecodeError, IndexError):
        return None


def calculate_total_score(scores):
    """Calculate total score from 4 dimensions."""
    if scores is None:
        return 0
    total = 0
    for k, v in scores.items():
        try:
            total += int(v)
        except (ValueError, TypeError):
            continue
    return total


def main():
    parser = argparse.ArgumentParser(description="WeStar Quality Scoring")
    parser.add_argument("--config", type=str, default="configs/scoring_config.yaml")
    parser.add_argument("--cluster_id", type=str, default=None)
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    cqsa_dir = config['data']['cqsa_dir']
    styles_path = config['data']['styles_path']
    comments_dir = config['data']['comments_dir']
    output_dir = config['output']['output_dir']
    
    min_total_score = config['filtering'].get('min_total_score', 19)  # out of 20
    top_k = config['filtering'].get('top_k', 10000)
    
    styles_data = {item['id']: item['style'] for item in load_jsonl(styles_path)}
    
    # Load scoring prompt
    with open(config['prompts']['quality_scoring'], 'r') as f:
        scoring_prompt = f.read()
    
    # Process clusters
    cluster_dirs = os.listdir(cqsa_dir)
    if args.cluster_id:
        cluster_dirs = [args.cluster_id]
    
    for cluster_id in cluster_dirs:
        cluster_path = os.path.join(cqsa_dir, cluster_id, "cqsa_data.jsonl")
        if not os.path.exists(cluster_path):
            continue
        
        print(f"\n=== Scoring Cluster {cluster_id} ===")
        cqsa_data = load_jsonl(cluster_path)
        
        scored_results = []
        for item in tqdm(cqsa_data, desc=f"Scoring cluster {cluster_id}"):
            biz_id = item.get('biz_id', '')
            style_labels = styles_data.get(biz_id, {})
            style_examples = ""  # Can add examples here
            
            scores = score_cqsa(item, style_labels, style_examples, scoring_prompt)
            total = calculate_total_score(scores)
            
            item['scores'] = scores
            item['total_score'] = total
            scored_results.append(item)
        
        # Filter and sort
        filtered = [item for item in scored_results if item['total_score'] >= min_total_score]
        filtered.sort(key=lambda x: x['total_score'], reverse=True)
        top_items = filtered[:top_k]
        
        # Save
        output_cluster_dir = os.path.join(output_dir, cluster_id)
        os.makedirs(output_cluster_dir, exist_ok=True)
        
        # Save all scored results
        save_jsonl(scored_results, os.path.join(output_cluster_dir, "metrics_mark_train_dpo.jsonl"))
        # Save filtered top results
        save_jsonl(top_items, os.path.join(output_cluster_dir, "cqsa_top.jsonl"))
        
        print(f"  Total: {len(scored_results)} | Filtered (≥{min_total_score}): {len(filtered)} | Top-{top_k}: {len(top_items)}")


if __name__ == "__main__":
    main()
