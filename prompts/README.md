# WeStar Prompts

本目录包含 WeStar 框架中使用的所有 Prompt 模板。

## 📋 Prompt 列表

| 文件 | 用途 | Pipeline 阶段 |
|------|------|---------------|
| `01_forward_question_gen.txt` | 正向问题生成（基于文章生成问题） | CQA 构造 |
| `02_bottom_up_role_gen.txt` | 自底向上用户角色生成 | CQA 构造 |
| `03_bottom_up_question_gen.txt` | 自底向上问题生成 | CQA 构造 |
| `04_answer_gen.txt` | 基于文章的答案生成 | CQA 构造 |
| `05_style_labeling.txt` | 单条 QA 风格标注（12维） | 风格标注 |
| `06_style_labeling_batch.txt` | 批量 QA 风格标注（简化版） | 风格标注 |
| `07_cqsa_rewriting.txt` | 风格化答案改写 | CQSA 构造 |
| `08_quality_scoring.txt` | CQSA 质量打分（4维评分） | 数据筛选 |
| `09_inference_prompt.txt` | 在线推理 prompt（含知识库） | 推理 |

## 🔄 变量说明

所有 prompt 中的变量使用 `{{variable_name}}` 格式，运行时会被实际内容替换。

| 变量 | 说明 |
|------|------|
| `{{title}}` | 文章标题 |
| `{{page_content}}` | 文章正文 |
| `{{name}}` | 公众号名称 |
| `{{domain}}` | 公众号领域 |
| `{{role}}` | 用户角色 |
| `{{biz_content}}` | 检索到的知识库文章内容 |
| `{{question}}` | 用户提问 |
| `{{answer}}` | 原始答案 |
| `{{twelve_labels}}` | 12维风格标签 |
| `{{examples}}` | 风格参考的 QA 示例 |
| `{{content}}` | 参考文章内容 |
| `{{style}}` | 作者风格描述 |
| `{{qas}}` | 风格参考的问答对 |
| `{{QAs}}` | 批量 QA 对 |
| `{{rewrite_answer}}` | 改写后的答案 |
