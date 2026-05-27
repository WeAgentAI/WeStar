"""
WeStar Step 1: CQA Data Construction
Generate Context-Question-Answer triplets using two strategies:
  - Forward-thinking: Generate questions from articles
  - Bottom-up: Simulate user roles → generate questions → retrieve context → generate answers
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


def forward_thinking_generate(articles, prompt_template):
    """Generate questions directly from article content."""
    results = []
    for article in tqdm(articles, desc="Forward-thinking CQ generation"):
        prompt = prompt_template.replace("{{title}}", article['title'])
        prompt = prompt_template.replace("{{page_content}}", article['page_content'])
        
        response = call_llm(prompt)
        try:
            parsed = json.loads(response.split('```json')[1].split('```')[0].strip())
            results.append({
                "question": parsed['question'],
                "context": article['page_content'],
                "title": article['title'],
                "source": "forward_thinking"
            })
        except (json.JSONDecodeError, IndexError):
            continue
    return results


def bottom_up_generate(biz_meta, articles, role_prompt_template, question_prompt_template):
    """Generate questions by simulating user roles."""
    results = []
    
    # Step 1: Generate roles and questions
    prompt = question_prompt_template.replace("{{name}}", biz_meta['name'])
    prompt = prompt.replace("{{domain}}", biz_meta['domain'])
    prompt = prompt.replace("{{role}}", biz_meta.get('role', '普通用户'))
    
    response = call_llm(prompt)
    try:
        parsed = json.loads(response.split('```json')[1].split('```')[0].strip())
        questions = parsed['question'] if isinstance(parsed['question'], list) else [parsed['question']]
    except (json.JSONDecodeError, IndexError):
        return results
    
    # Step 2: For each question, retrieve relevant articles and generate answers
    for question in questions:
        # Simple retrieval: find most relevant article (can be replaced with vector search)
        relevant_articles = retrieve_articles(question, articles, top_k=3)
        context = "\n".join([a['page_content'] for a in relevant_articles])
        
        results.append({
            "question": question,
            "context": context,
            "source": "bottom_up"
        })
    
    return results


def retrieve_articles(question, articles, top_k=3):
    """Simple keyword-based retrieval. Replace with vector search in production."""
    # Placeholder: return first top_k articles
    # In production, use embedding-based retrieval
    return articles[:top_k]


def generate_answers(cq_pairs, answer_prompt_path):
    """Generate answers for each CQ pair."""
    with open(answer_prompt_path, 'r') as f:
        answer_template = f.read()
    
    results = []
    for pair in tqdm(cq_pairs, desc="Answer generation"):
        prompt = answer_template.replace("{{biz_content}}", pair['context'])
        prompt = prompt.replace("{{question}}", pair['question'])
        
        answer = call_llm(prompt)
        pair['answer'] = answer
        results.append(pair)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="WeStar CQA Data Construction")
    parser.add_argument("--config", type=str, default="configs/cqa_config.yaml")
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    # Load data
    articles_dir = config['data']['articles_dir']
    biz_meta_path = config['data']['biz_meta_path']
    output_dir = config['output']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    with open(biz_meta_path, 'r') as f:
        biz_meta_list = json.load(f)
    
    # Load prompt templates
    with open(config['prompts']['forward_question'], 'r') as f:
        forward_prompt = f.read()
    with open(config['prompts']['bottom_up_question'], 'r') as f:
        bottom_up_prompt = f.read()
    
    all_cqa = []
    
    for biz_meta in tqdm(biz_meta_list, desc="Processing accounts"):
        biz_id = biz_meta['biz_id']
        articles_path = os.path.join(articles_dir, f"{biz_id}.jsonl")
        
        if not os.path.exists(articles_path):
            continue
        
        articles = load_jsonl(articles_path)
        
        # Strategy 1: Forward-thinking
        forward_cq = forward_thinking_generate(articles, forward_prompt)
        
        # Strategy 2: Bottom-up
        bottom_up_cq = bottom_up_generate(biz_meta, articles, "", bottom_up_prompt)
        
        # Combine and generate answers
        all_cq = forward_cq + bottom_up_cq
        cqa_data = generate_answers(all_cq, config['prompts']['answer_gen'])
        
        # Add biz_id
        for item in cqa_data:
            item['biz_id'] = biz_id
        
        all_cqa.extend(cqa_data)
    
    # Save
    output_path = os.path.join(output_dir, "cqa_data.jsonl")
    save_jsonl(all_cqa, output_path)
    print(f"Generated {len(all_cqa)} CQA instances → {output_path}")


if __name__ == "__main__":
    main()
