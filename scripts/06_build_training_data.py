"""
WeStar Step 6: Build Training Data
Construct training data for three training paradigms:
  - SFT+: Top-scored CQSA as SFT data (ShareGPT format)
  - SeDPO: Style-Enhanced DPO with sibling cluster rejected samples
  - MDPO: Metric-guided DPO with base model distillates as rejected samples
"""
import json
import os
import argparse
import yaml
from tqdm import tqdm
from utils import load_jsonl, save_jsonl


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


# ============= System Prompts =============

SYSTEM_PROMPT_WITH_CONTEXT = '''# 你是一个智能助手。你要把自己想象成账号作者本人。回复时不要出现"（参考xxx)"字样。你能够结合[知识库]中账号的历史文章内容回复用户提供的各种问题。你在回复的时候要参考[知识库]中的历史文章的内容，并根据[历史留言知识库]中作者回复留言的说话语气来回复用户，但回答中不要引用任何历史留言的内容。现在，你需要遵循以下原则，与你的关注者（或称为用户）进行友善、有价值的沟通和互动。回复时不要出现"（参考xxx)"字样。'''

SYSTEM_PROMPT_CHAT = '''# 你是一个智能助手。你要把自己想象成账号作者本人。用户的意图是闲聊。现在，你需要遵循以下原则，与你的关注者（或称为用户）进行友善、有价值的沟通和互动。回复时不要出现"（参考xxx)"字样。'''


def build_input_with_context(content, question):
    """Build input prompt with knowledge base context."""
    return f'''#从历史文章中检索出来的结果如下：
[知识库]="
{content}"
- 我给你的检索结果中，每一篇单独的文章都是按照"content"的格式，其中content为对应文章的内容。但注意回答时不要通过括号注明引用了哪篇文章。
- 你需要模仿检索出的文章中作者的语气和风格来回复用户的问题.
- 不要显式引用任何文章，回答不要出现圆括号，不要出现"（参考历史文章xxx)"字样。
- 如果知识库返回为空，你的回复中需要解释一下你没写过类似的文章，然后凭借你自己的经验给出回复，注意，仅对[知识库]内容做总结，不要显式地引用文章，如果知识库返回结果为空，不要参考或引用任何的文章

# 回复表达方式的要求如下：
- 不要输出空的书名号
- 结合用户聊天历史回答问题，回答简洁，避免科普式长篇大论：回复控制在 150 字内，避免长篇大论与信息堆砌。若内容复杂，可拆分成多段。
- 严格基于[知识库]提供的信息和写作⻛格。
- a.专业性问题深入解答：面对专业问题，基于[知识库]的内容提供专业准确的回答，核心观点保持一致。不了解的内容，请诚实地告知用户，不要编造事实。;b.闲聊话题轻松沟通：可多用口语化的表达方式，不可以使用感叹号，例如"！"。请始终保持善良、正直、友好、活泼、口语化的交互风格。;c.与用户平等对话：不要谄媚或讨好，也不能傲慢无礼，保持平等、尊重的交流。;
- 不要输出markdown格式的文本，例如"**标题**"之类的文字。
- 不要滥用圆括号"（）"补充内容，例如："字体打架像菜市场吵架（笑）"。
#用户提问的原始问题为：{question}
根据要求和用户问题，你的回答为：'''


def build_input_chat_only(question):
    """Build input prompt for chat-only (no context)."""
    return f'''# 回复表达方式的要求如下：
- 不要输出空的书名号
- 回答简洁，避免科普式长篇大论：回复控制在 150 字内，避免长篇大论与信息堆砌。若内容复杂，可拆分成多段。
- a.专业性问题深入解答：面对专业问题，基于[知识库]的内容提供专业准确的回答，核心观点保持一致。不了解的内容，请诚实地告知用户，不要编造事实。;b.闲聊话题轻松沟通：可多用口语化的表达方式，不可以使用感叹号，例如"！"。请始终保持善良、正直、友好、活泼、口语化的交互风格。;c.与用户平等对话：不要谄媚或讨好，也不能傲慢无礼，保持平等、尊重的交流。;
- 不要输出markdown格式的文本，例如"**标题**"之类的文字。
- 不要滥用圆括号"（）"补充内容，例如："字体打架像菜市场吵架（笑）"。
#用户提问的原始问题为：{question}
根据要求和用户问题，你的回答为：'''


def build_sft_instance(origin_json):
    """Build a SFT training instance in ShareGPT format."""
    if 'content' not in origin_json or not origin_json.get('content'):
        system_prompt = SYSTEM_PROMPT_CHAT
        user_input = build_input_chat_only(origin_json['question'])
    else:
        system_prompt = SYSTEM_PROMPT_WITH_CONTEXT
        user_input = build_input_with_context(origin_json['content'], origin_json['question'])
    
    return {
        "system": system_prompt,
        "conversations": [
            {"from": "human", "value": user_input},
            {"from": "gpt", "value": origin_json['rewrite_answer']}
        ]
    }


def build_dpo_instance(origin_json, rejected_answer):
    """Build a DPO training instance in ShareGPT format."""
    if 'content' not in origin_json or not origin_json.get('content'):
        system_prompt = SYSTEM_PROMPT_CHAT
        user_input = build_input_chat_only(origin_json['question'])
    else:
        system_prompt = SYSTEM_PROMPT_WITH_CONTEXT
        user_input = build_input_with_context(origin_json['content'], origin_json['question'])
    
    return {
        "conversations": [
            {"from": "system", "value": system_prompt},
            {"from": "human", "value": user_input}
        ],
        "chosen": {"from": "gpt", "value": origin_json['rewrite_answer']},
        "rejected": {"from": "gpt", "value": rejected_answer}
    }


def build_sft_data(scored_path, min_score=19):
    """Build SFT+ training data from top-scored samples."""
    results = []
    
    with open(scored_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            try:
                if '</think>' in item.get('inference_result', ''):
                    scores_str = item['inference_result'].split('</think>')[1].strip()
                else:
                    scores_str = item.get('inference_result', '{}')
                scores_str = scores_str.split('```json')[1].split('```')[0].strip()
                scores = json.loads(scores_str)
            except (json.JSONDecodeError, IndexError):
                continue
            
            total = sum(int(v) for v in scores.values())
            if total < min_score:
                continue
            
            origin_json = item['origin_json']
            sft_instance = build_sft_instance(origin_json)
            results.append(sft_instance)
    
    # Sort by total length (descending)
    results.sort(
        key=lambda x: len(x['conversations'][0]['value']) + len(x['conversations'][1]['value']),
        reverse=True
    )
    return results


def build_sedpo_data(chosen_scored_path, rejected_scored_path, min_score=19):
    """
    Build SeDPO training data.
    Chosen: top-scored CQSA from target cluster
    Rejected: answers from sibling cluster (same question, different style)
    """
    # Load rejected answers indexed by question
    reject_map = {}
    with open(rejected_scored_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            origin = item.get('origin_json', {})
            if 'question' in origin and 'rewrite_answer' in origin:
                reject_map[origin['question']] = origin['rewrite_answer']
    
    # Build DPO pairs
    results = []
    with open(chosen_scored_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            try:
                if '</think>' in item.get('inference_result', ''):
                    scores_str = item['inference_result'].split('</think>')[1].strip()
                else:
                    scores_str = item.get('inference_result', '{}')
                scores_str = scores_str.split('```json')[1].split('```')[0].strip()
                scores = json.loads(scores_str)
            except (json.JSONDecodeError, IndexError):
                continue
            
            total = sum(int(v) for v in scores.values())
            if total < min_score:
                continue
            
            origin_json = item['origin_json']
            question = origin_json['question']
            
            # SeDPO: use sibling cluster's answer as rejected
            if question in reject_map:
                rejected_answer = reject_map[question]
            else:
                continue  # Skip if no matching rejected sample
            
            dpo_instance = build_dpo_instance(origin_json, rejected_answer)
            results.append(dpo_instance)
    
    # Sort by total length
    results.sort(
        key=lambda x: len(x['conversations'][0]['value']) + len(x['conversations'][1]['value']) + len(x['chosen']['value']),
        reverse=True
    )
    return results


def build_mdpo_data(scored_path, min_score=19):
    """
    Build MDPO training data.
    Chosen: stylized rewritten answer
    Rejected: original (non-stylized) answer
    """
    results = []
    with open(scored_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            try:
                if '</think>' in item.get('inference_result', ''):
                    scores_str = item['inference_result'].split('</think>')[1].strip()
                else:
                    scores_str = item.get('inference_result', '{}')
                scores_str = scores_str.split('```json')[1].split('```')[0].strip()
                scores = json.loads(scores_str)
            except (json.JSONDecodeError, IndexError):
                continue
            
            total = sum(int(v) for v in scores.values())
            if total < min_score:
                continue
            
            origin_json = item['origin_json']
            rejected_answer = origin_json.get('answer', '')  # Original non-stylized answer
            
            dpo_instance = build_dpo_instance(origin_json, rejected_answer)
            results.append(dpo_instance)
    
    results.sort(
        key=lambda x: len(x['conversations'][0]['value']) + len(x['conversations'][1]['value']) + len(x['chosen']['value']),
        reverse=True
    )
    return results


def main():
    parser = argparse.ArgumentParser(description="WeStar Training Data Construction")
    parser.add_argument("--config", type=str, default="configs/training_config.yaml")
    parser.add_argument("--cluster_id", type=str, required=True, help="Target cluster ID")
    parser.add_argument("--mode", type=str, choices=['sft', 'sedpo', 'mdpo', 'all'], default='all')
    parser.add_argument("--sibling_cluster_id", type=str, default=None, 
                       help="Sibling cluster ID for SeDPO rejected samples")
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    cluster_dir = os.path.join(config['data']['cluster_tree_dir'], args.cluster_id)
    scored_path = os.path.join(cluster_dir, "metrics_mark_train_dpo.jsonl")
    min_score = config['filtering'].get('min_total_score', 19)
    
    if args.mode in ['sft', 'all']:
        print(f"\n=== Building SFT+ data for cluster {args.cluster_id} ===")
        sft_data = build_sft_data(scored_path, min_score)
        output_path = os.path.join(cluster_dir, "train_lora_top_sft.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(sft_data, f, ensure_ascii=False, indent=4)
        print(f"  SFT+ instances: {len(sft_data)} → {output_path}")
    
    if args.mode in ['sedpo', 'all']:
        if args.sibling_cluster_id is None:
            print("  [WARN] Skipping SeDPO: --sibling_cluster_id not provided")
        else:
            print(f"\n=== Building SeDPO data (cluster {args.cluster_id} vs {args.sibling_cluster_id}) ===")
            sibling_dir = os.path.join(config['data']['cluster_tree_dir'], args.sibling_cluster_id)
            rejected_path = os.path.join(sibling_dir, "metrics_mark_train_dpo.jsonl")
            
            sedpo_data = build_sedpo_data(scored_path, rejected_path, min_score)
            output_path = os.path.join(cluster_dir, "train_lora_style_dpo.json")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(sedpo_data, f, ensure_ascii=False, indent=4)
            print(f"  SeDPO instances: {len(sedpo_data)} → {output_path}")
    
    if args.mode in ['mdpo', 'all']:
        print(f"\n=== Building MDPO data for cluster {args.cluster_id} ===")
        mdpo_data = build_mdpo_data(scored_path, min_score)
        output_path = os.path.join(cluster_dir, "train_lora_dpo.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(mdpo_data, f, ensure_ascii=False, indent=4)
        print(f"  MDPO instances: {len(mdpo_data)} → {output_path}")


if __name__ == "__main__":
    main()
