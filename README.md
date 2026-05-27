# WeStar: One Agent to Serve All

> A Lite-Adaptive Stylized AI Assistant for Millions of Multi-Style Official Accounts

[![ACL 2026 Findings](https://img.shields.io/badge/ACL%202026-Findings-orange)](https://arxiv.org/abs/2509.17788)
[![arXiv](https://img.shields.io/badge/arXiv-2509.17788-b31b1b.svg)](https://arxiv.org/abs/2509.17788)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📖 Overview

**WeStar** is a unified framework for stylized contextual question answering (CQSA) deployed on large-scale content platforms. It serves millions of accounts with a single model by combining:

- **Knowledge injection** via prompt (retrieved articles)
- **Style injection** via LoRA parameters (per-cluster style adaptation)

<p align="center">
  <img src="assets/framework.png" width="80%" alt="WeStar Framework"/>
</p>

## 🏗️ Architecture

```
WeStar Pipeline:
┌──────────────┐     ┌──────────────┐     ┌──────────────────┐
│  CQA Data    │ ──▶ │ Style Label  │ ──▶ │ Style Tree Build │
│ Construction │     │   Annotate   │     │   & Clustering   │
└──────────────┘     └──────────────┘     └──────────────────┘
                                                    │
┌──────────────┐     ┌──────────────┐               ▼
│   Online     │ ◀── │   SeDPO      │ ◀── ┌──────────────────┐
│  Inference   │     │   Training   │     │  CQSA Rewriting  │
└──────────────┘     └──────────────┘     │ + Quality Filter │
                                          └──────────────────┘
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone the repo
git clone https://github.com/user/WeStar.git
cd WeStar

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation

Prepare your data in the following structure:

```
data/
├── articles/              # Historical articles per account (jsonl)
│   └── {biz_id}.jsonl     # One file per account
├── comments/              # User comments & author replies (jsonl)
│   └── {biz_id}.jsonl
└── biz_meta.json          # Account metadata (name, domain, etc.)
```

**Data Format:**

```jsonl
# articles/{biz_id}.jsonl
{"title": "文章标题", "page_content": "文章正文内容..."}

# comments/{biz_id}.jsonl  
{"question": "用户留言内容", "answer": "作者回复内容"}
```

### 3. Run the Full Pipeline

```bash
# Step 1: Generate CQA data (forward-thinking + bottom-up)
python scripts/01_generate_cqa.py --config configs/cqa_config.yaml

# Step 2: Style labeling for all accounts
python scripts/02_style_labeling.py --config configs/style_config.yaml

# Step 3: Build style clustering tree
python scripts/03_build_style_tree.py --config configs/tree_config.yaml

# Step 4: CQSA stylized rewriting
python scripts/04_cqsa_rewriting.py --config configs/rewrite_config.yaml

# Step 5: Quality scoring & filtering
python scripts/05_quality_scoring.py --config configs/scoring_config.yaml

# Step 6: Build training data (SFT / SeDPO / MDPO)
python scripts/06_build_training_data.py --config configs/training_config.yaml

# Step 7: Train LoRA per cluster (using LLaMA-Factory / similar)
bash scripts/07_train_lora.sh --cluster_id 0

# Step 8: Online inference
python scripts/08_inference.py --question "你好" --biz_id "xxx"
```

### 4. Quick Demo (Single Account)

```bash
# Run end-to-end for a single account
python scripts/run_single_account.py \
    --biz_id "your_biz_id" \
    --articles_path data/articles/your_biz_id.jsonl \
    --comments_path data/comments/your_biz_id.jsonl \
    --output_dir output/your_biz_id/
```

## 📁 Project Structure

```
WeStar/
├── README.md                  # This file
├── requirements.txt           # Python dependencies
├── configs/                   # Configuration files
│   ├── cqa_config.yaml
│   ├── style_config.yaml
│   ├── tree_config.yaml
│   ├── rewrite_config.yaml
│   ├── scoring_config.yaml
│   └── training_config.yaml
├── prompts/                   # All prompt templates
│   ├── README.md              # Prompt documentation
│   ├── 01_forward_question_gen.txt
│   ├── 02_bottom_up_role_gen.txt
│   ├── 03_bottom_up_question_gen.txt
│   ├── 04_answer_gen.txt
│   ├── 05_style_labeling.txt
│   ├── 06_style_labeling_batch.txt
│   ├── 07_cqsa_rewriting.txt
│   ├── 08_quality_scoring.txt
│   └── 09_inference_prompt.txt
├── scripts/                   # Pipeline scripts
│   ├── 01_generate_cqa.py
│   ├── 02_style_labeling.py
│   ├── 03_build_style_tree.py
│   ├── 04_cqsa_rewriting.py
│   ├── 05_quality_scoring.py
│   ├── 06_build_training_data.py
│   ├── 07_train_lora.sh
│   ├── 08_inference.py
│   └── utils.py
├── data/                      # Data directory (gitignored)
│   └── .gitkeep
└── assets/                    # Images for README
    └── framework.png
```

## 📐 Style Dimensions (12D)

WeStar uses 12 style dimensions for annotation:

| Dimension | Description | Example Labels |
|-----------|-------------|---------------|
| Lexical Complexity | 词汇复杂度 | 简单口语化 / 专业术语化 |
| Emotional Tendency | 情感倾向 | 积极情感 / 中性客观 / 消极安抚 |
| Degree of Formalization | 形式化程度 | 高正式度 / 亲切随意 |
| Sentence Complexity | 句式复杂度 | 简洁短句 / 复合长句 |
| Rhetorical Features | 修辞特征 | 直白陈述 / 比喻类比 / 排比强调 |
| Connection Mechanism | 衔接机制 | 强逻辑衔接 / 碎片化表达 |
| Omitted Features | 省略特征 | 高频主语省略 / 句子完整 |
| Inversion Sentence | 倒装使用 | 常规语序 / 情感倒装 / 逻辑倒装 |
| Passive Sentences | 被动语态 | 显性被动 / 隐性被动 / 不使用 |
| Authority | 权威程度 | 高权威性 / 协商建议 |
| Positivity | 肯定性 | 高肯定性 / 高否定性 / 不明显 |
| Emoji Frequency | 表情使用频率 | 高频 / 仅简短回复 / 不使用 |

## 🔧 Training Details

| Component | Setting |
|-----------|---------|
| Base Model | Qwen3-32B |
| Auxiliary LLM (M) | DeepSeek-R1 |
| Fine-tuning | LoRA (per style cluster) |
| SFT+ data | Top-scored CQSA (score ≥ 19/20) |
| SeDPO chosen | Top CQSA from target cluster |
| SeDPO rejected | Same question, sibling cluster answer |
| Eval Judge | DeepSeek-R1 |

## 📊 Results

### Automatic Evaluation (Average across 10 clusters)

| Method | Q–A | C–A | S–A | Fluency |
|--------|-----|-----|-----|---------|
| R1-Prompt | 4.38 | 4.45 | 4.25 | 4.75 |
| SFT-Prompt | 4.26 | 4.30 | 3.73 | 4.70 |
| LoRA-SFT | 4.35 | 4.43 | 3.92 | 4.73 |
| LoRA-SFT-S | 4.41 | 4.49 | 4.22 | 4.77 |
| WeStar_MDPO | **4.44** | 4.52 | 4.20 | 4.76 |
| **WeStar** | 4.43 | **4.55** | **4.25** | **4.77** |

## 📝 Citation

If you find this work helpful, please cite our paper:

```bibtex
@inproceedings{fan2026westar,
  title={One Agent to Serve All: a Lite-Adaptive Stylized AI Assistant for Millions of Multi-Style Official Accounts},
  author={Fan, Xingyu and Li, Feifei and Que, Wenhui and Li, Hailong},
  booktitle={Findings of the Association for Computational Linguistics: ACL 2026},
  year={2026},
  url={https://arxiv.org/abs/2509.17788}
}
```

📄 **Paper**: [https://arxiv.org/abs/2509.17788](https://arxiv.org/abs/2509.17788)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
