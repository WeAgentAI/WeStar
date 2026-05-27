"""
WeStar Step 2: Style Labeling
Annotate each author's comment-reply pairs with 12-dimensional style labels,
then extract the mode (majority vote) for each dimension as the account's style profile.
"""
import json
import os
import argparse
import yaml
from tqdm import tqdm
from collections import Counter
from utils import call_llm, load_jsonl, save_jsonl


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


STYLE_DIMENSIONS = [
    'lexical complexity', 'Emotional tendency', 'Degree of formalization',
    'Sentence complexity', 'Rhetorical Features', 'Connection Mechanism',
    'Omitted Features', 'Inversion Sentence', 'Passive Sentences',
    'Authority', 'Positivity', 'Emoji Frequency'
]


def label_single_qa(question, answer, prompt_template):
    """Label a single QA pair with 12D style dimensions."""
    prompt = prompt_template.replace("{{question}}", question)
    prompt = prompt.replace("{{answer}}", answer)
    
    response = call_llm(prompt)
    try:
        result = response.split('```json')[1].split('```')[0].strip()
        return json.loads(result)
    except (json.JSONDecodeError, IndexError):
        return None


def label_batch_qa(qas, prompt_template):
    """Label a batch of QA pairs (simplified version for annotation platform)."""
    qa_text = "\n".join([f"问题：{qa['question']}\n回答：{qa['answer']}" for qa in qas])
    prompt = prompt_template.replace("{{QAs}}", qa_text)
    
    response = call_llm(prompt)
    try:
        result = response.split('```json')[1].split('```')[0].strip()
        return json.loads(result)
    except (json.JSONDecodeError, IndexError):
        return None


def extract_mode_labels(label_results):
    """Extract mode (majority vote) label for each style dimension."""
    style_profile = {}
    
    for key in STYLE_DIMENSIONS:
        values = []
        for result in label_results:
            if result is None or key not in result:
                continue
            value = result[key]
            # Clean up label value
            if '.' in value:
                value = value.split('.')[1].strip()
            if ':' in value:
                value = value.split(':')[0].strip()
            if '：' in value:
                value = value.split('：')[0].strip()
            if '(' in value:
                value = value.split('(')[0].strip()
            if '（' in value:
                value = value.split('（')[0].strip()
            values.append(value)
        
        if values:
            counter = Counter(values)
            style_profile[key] = counter.most_common(1)[0][0]
        else:
            style_profile[key] = "N/A"
    
    return style_profile


def main():
    parser = argparse.ArgumentParser(description="WeStar Style Labeling")
    parser.add_argument("--config", type=str, default="configs/style_config.yaml")
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    comments_dir = config['data']['comments_dir']
    output_path = config['output']['styles_path']
    use_batch = config.get('use_batch_labeling', False)
    
    # Load prompt template
    if use_batch:
        with open(config['prompts']['style_labeling_batch'], 'r') as f:
            prompt_template = f.read()
    else:
        with open(config['prompts']['style_labeling'], 'r') as f:
            prompt_template = f.read()
    
    # Process each account
    results = []
    
    comment_files = [f for f in os.listdir(comments_dir) if f.endswith('.jsonl')]
    
    for filename in tqdm(comment_files, desc="Style labeling"):
        biz_id = filename.replace('.jsonl', '')
        comments = load_jsonl(os.path.join(comments_dir, filename))
        
        if use_batch:
            # Batch mode: send all QAs at once
            label_result = label_batch_qa(comments, prompt_template)
            style_profile = label_result if label_result else {}
        else:
            # Single mode: label each QA individually then extract mode
            label_results = []
            for comment in comments:
                result = label_single_qa(
                    comment['question'],
                    comment['answer'],
                    prompt_template
                )
                label_results.append(result)
            
            style_profile = extract_mode_labels(label_results)
        
        results.append({
            "id": biz_id,
            "style": style_profile
        })
    
    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_jsonl(results, output_path)
    print(f"Labeled {len(results)} accounts → {output_path}")


if __name__ == "__main__":
    main()
