"""
WeStar Step 8: Online Inference
Perform style-aware contextual QA by combining:
  - Knowledge injection via prompt (retrieved articles)
  - Style injection via LoRA parameters (per-cluster adapter)
"""
import json
import os
import argparse
import time
import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class WeStarInference:
    """WeStar Online Inference Engine."""
    
    def __init__(self, base_model_path, label_path, lora_dir, device="cuda"):
        self.device = device
        
        # Load base model and tokenizer
        print("Loading base model...")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True
        )
        
        # Load cluster labels
        with open(label_path, 'r') as f:
            self.label_map = json.load(f)
        
        # LoRA directory
        self.lora_dir = lora_dir
        self.current_lora = None
        
        # Load inference prompt template
        prompt_path = os.path.join(os.path.dirname(__file__), '..', 'prompts', '09_inference_prompt.txt')
        with open(prompt_path, 'r') as f:
            self.inference_prompt = f.read()
    
    def get_cluster_id(self, biz_id):
        """Look up the style cluster for a given account."""
        return str(self.label_map.get(biz_id, 0))
    
    def load_lora(self, cluster_id):
        """Load the LoRA adapter for the target cluster."""
        if self.current_lora == cluster_id:
            return  # Already loaded
        
        lora_path = os.path.join(self.lora_dir, f"cluster_{cluster_id}_dpo")
        if not os.path.exists(lora_path):
            lora_path = os.path.join(self.lora_dir, f"cluster_{cluster_id}_top_sft")
        
        if os.path.exists(lora_path):
            # Unload previous LoRA if any
            if self.current_lora is not None:
                self.model = self.model.base_model
            
            self.model = PeftModel.from_pretrained(self.model, lora_path)
            self.current_lora = cluster_id
            print(f"  Loaded LoRA for cluster {cluster_id}")
        else:
            print(f"  [WARN] No LoRA found for cluster {cluster_id}, using base model")
    
    def build_prompt(self, question, context, style_desc="", style_examples=""):
        """Build the inference prompt with knowledge and style context."""
        prompt = self.inference_prompt
        prompt = prompt.replace("{{content}}", context)
        prompt = prompt.replace("{{question}}", question)
        prompt = prompt.replace("{{style}}", style_desc)
        prompt = prompt.replace("{{qas}}", style_examples)
        return prompt
    
    def generate(self, biz_id, question, context, style_desc="", style_examples="",
                 max_new_tokens=512, temperature=0.7):
        """
        Generate a stylized response.
        
        Args:
            biz_id: Account ID for style lookup
            question: User's question
            context: Retrieved article context
            style_desc: Style description (12D labels)
            style_examples: Example QA pairs for style reference
        
        Returns:
            dict with response and timing info
        """
        t_start = time.time()
        
        # Step 1: Style tree lookup
        t_lookup_start = time.time()
        cluster_id = self.get_cluster_id(biz_id)
        t_lookup = time.time() - t_lookup_start
        
        # Step 2: Load LoRA
        t_lora_start = time.time()
        self.load_lora(cluster_id)
        t_lora = time.time() - t_lora_start
        
        # Step 3: Build prompt
        prompt = self.build_prompt(question, context, style_desc, style_examples)
        
        # Step 4: LLM inference
        t_infer_start = time.time()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9
        )
        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        t_infer = time.time() - t_infer_start
        
        t_total = time.time() - t_start
        
        return {
            "response": response,
            "cluster_id": cluster_id,
            "timing": {
                "style_lookup": f"{t_lookup:.3f}s",
                "lora_loading": f"{t_lora:.3f}s",
                "llm_inference": f"{t_infer:.3f}s",
                "total": f"{t_total:.3f}s"
            }
        }


def main():
    parser = argparse.ArgumentParser(description="WeStar Online Inference")
    parser.add_argument("--base_model", type=str, required=True, help="Path to base model")
    parser.add_argument("--label_path", type=str, default="cluster_tree/label.json")
    parser.add_argument("--lora_dir", type=str, default="checkpoints/")
    parser.add_argument("--biz_id", type=str, required=True, help="Target account ID")
    parser.add_argument("--question", type=str, required=True, help="User question")
    parser.add_argument("--context", type=str, default="", help="Retrieved article context")
    parser.add_argument("--style_desc", type=str, default="", help="Style description JSON")
    parser.add_argument("--style_examples", type=str, default="", help="Style example QAs")
    args = parser.parse_args()
    
    # Initialize engine
    engine = WeStarInference(
        base_model_path=args.base_model,
        label_path=args.label_path,
        lora_dir=args.lora_dir
    )
    
    # Generate response
    result = engine.generate(
        biz_id=args.biz_id,
        question=args.question,
        context=args.context,
        style_desc=args.style_desc,
        style_examples=args.style_examples
    )
    
    print(f"\n{'='*50}")
    print(f"Question: {args.question}")
    print(f"Cluster: {result['cluster_id']}")
    print(f"{'='*50}")
    print(f"\nResponse:\n{result['response']}")
    print(f"\nTiming: {result['timing']}")


if __name__ == "__main__":
    main()
