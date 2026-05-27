"""
WeStar Step 3: Build Style Clustering Tree
Group accounts with similar style into clusters using hierarchical partitioning
based on selected style dimensions.
"""
import json
import os
import argparse
import yaml
from utils import load_jsonl


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def build_style_tree(styles_data, cluster_metrics, min_cluster_size=100):
    """
    Build hierarchical style tree by binary encoding based on style dimensions.
    
    Args:
        styles_data: list of {"id": biz_id, "style": {...}}
        cluster_metrics: dict of {dimension: positive_value} for clustering
        min_cluster_size: minimum number of samples in a cluster
    
    Returns:
        label_map: dict of {biz_id: cluster_label}
    """
    label_map = {}
    
    # Step 1: Encode each account as binary label based on cluster_metrics
    for item in styles_data:
        biz_id = item['id']
        style = item['style']
        
        if isinstance(style, str):
            try:
                style = json.loads(style.replace("'", '"'))
            except json.JSONDecodeError:
                continue
        
        label = 0
        for metric, positive_value in cluster_metrics.items():
            if style.get(metric) == positive_value:
                label = label * 2 + 1
            else:
                label = label * 2
        
        label_map[biz_id] = label
    
    # Step 2: Count cluster sizes
    cluster_counts = {}
    for biz_id, cluster_id in label_map.items():
        cluster_counts[cluster_id] = cluster_counts.get(cluster_id, 0) + 1
    
    # Step 3: Merge small clusters into parent
    max_label = max(cluster_counts.keys()) if cluster_counts else 0
    for cluster_id in range(max_label, -1, -1):
        if cluster_id not in cluster_counts:
            continue
        if cluster_counts[cluster_id] < min_cluster_size:
            parent_id = cluster_id // 2
            if parent_id not in cluster_counts:
                cluster_counts[parent_id] = 0
            cluster_counts[parent_id] += cluster_counts[cluster_id]
            
            # Reassign all accounts in this cluster to parent
            for biz_id in label_map:
                if label_map[biz_id] == cluster_id:
                    label_map[biz_id] = parent_id
            
            cluster_counts[cluster_id] = 0
    
    # Remove empty clusters
    cluster_counts = {k: v for k, v in cluster_counts.items() if v > 0}
    
    return label_map, cluster_counts


def main():
    parser = argparse.ArgumentParser(description="WeStar Style Tree Building")
    parser.add_argument("--config", type=str, default="configs/tree_config.yaml")
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    # Load style data
    styles_path = config['data']['styles_path']
    styles_data = load_jsonl(styles_path)
    
    # Clustering dimensions and their "positive" values
    # These determine how the binary tree is split
    cluster_metrics = config['clustering']['dimensions']
    # Example:
    # cluster_metrics = {
    #     'Authority': '高权威性',
    #     'Emotional tendency': '积极情感',
    #     'Positivity': '肯定性与否定性不明显',
    #     'Emoji Frequency': '高使用频率'
    # }
    
    min_cluster_size = config['clustering'].get('min_cluster_size', 100)
    
    # Build tree
    label_map, cluster_counts = build_style_tree(
        styles_data, cluster_metrics, min_cluster_size
    )
    
    # Output
    output_dir = config['output']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    # Save label mapping
    label_path = os.path.join(output_dir, "label.json")
    with open(label_path, 'w') as f:
        json.dump(label_map, f, ensure_ascii=False, indent=4)
    
    # Create directories for each cluster
    for cluster_id in cluster_counts:
        cluster_dir = os.path.join(output_dir, str(cluster_id))
        os.makedirs(cluster_dir, exist_ok=True)
    
    # Print summary
    print(f"\n=== Style Tree Summary ===")
    print(f"Total accounts: {len(label_map)}")
    print(f"Number of clusters: {len(cluster_counts)}")
    print(f"\nCluster sizes:")
    for cluster_id, count in sorted(cluster_counts.items()):
        print(f"  Cluster {cluster_id}: {count} accounts")
    print(f"\nSaved to: {label_path}")


if __name__ == "__main__":
    main()
