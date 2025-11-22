import json
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import math


def load_data(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def parse_task_name(name):
    parts = name.split("_")
    if name.startswith("random_order") and len(parts) == 3:
        return "random", None, parts[-1]
    if name.startswith("reorder") and len(parts) == 3:
        return "reorder", parts[1], parts[2]
    return None, None, None


def group_by_target(data):
    groups = defaultdict(list)
    for item in data:
        name = item["training_remark"]
        mode, src, tgt = parse_task_name(name)
        if mode is None:
            continue
        groups[tgt].append({
            "mode": mode,
            "source": src,
            "metrics": item,
            "name": name
        })
    return groups


def extract_metric_fields(data):
    exclude_fields = {"test_runtime", "test_samples_per_second", "test_steps_per_second", "test_file"}
    example = data[0]
    return [k for k in example.keys() if k.startswith("test_") and k not in exclude_fields]


def plot_target_grid(target_name, items, metric_fields, max_cols=4):

    light_colors = [
        "#aec6cf", "#ffb347", "#c3e6cb", "#f7c6c7",
        "#e3d7ff", "#fff3b0", "#d4f0f0", "#ffd6e0"
    ]

    n_metrics = len(metric_fields)
    n_cols = min(max_cols, n_metrics)
    n_rows = math.ceil(n_metrics / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*4, n_rows*3), constrained_layout=True)

    axes = axes.flatten() if n_metrics > 1 else [axes]

    for idx, metric in enumerate(metric_fields):
        ax = axes[idx]
        labels, values, colors = [], [], []

        for i, item in enumerate(items):
            mode, src = item["mode"], item["source"]
            val = item["metrics"].get(metric, None)
            if val is None:
                continue
            label = f"{src}→{target_name}" if mode=="reorder" else f"random({target_name})"
            color = light_colors[i % len(light_colors)]
            labels.append(label)
            values.append(val)
            colors.append(color)

        x = np.arange(len(labels))
        bars = ax.bar(x, values, color=colors)

        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h, f"{h:.3f}", ha='center', va='bottom', fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.set_title(metric, fontsize=10, fontweight='bold')

    
    for j in range(n_metrics, n_rows*n_cols):
        fig.delaxes(axes[j])

    fig.suptitle(f"All Metrics for Target: {target_name}", fontsize=14, fontweight='bold')
    plt.show()


if __name__ == "__main__":
    path = "./rq4_knowledge_inteference_metrics_v3.jsonl"
    data = load_data(path)
    groups = group_by_target(data)
    metric_fields = extract_metric_fields(data)
    print("Metrics detected:", metric_fields)

    # target_name = list(groups.keys())[0]
    # items = groups[target_name]
    # plot_target_grid(target_name, items, metric_fields)

    for target_name, items in groups.items():
        plot_target_grid(target_name, items, metric_fields)




