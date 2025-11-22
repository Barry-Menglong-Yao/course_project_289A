#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import random
from collections import OrderedDict
import argparse
from templates_mc import relation_templates

def generate_qa(data_item, same_relation_data, relation_templates):
    """
    data_item
    same_relation_data:
    relation_templates
    """

    templates = relation_templates.get(data_item['relation'], relation_templates.get("Synonym"))
    template = random.choice(templates)

    current_concept = data_item['concept']

    distractor_pool = list({d['object'] for d in same_relation_data if d['concept'] != current_concept})

    if len(distractor_pool) >= 3:
        distractors = random.sample(distractor_pool, 3)
    elif len(distractor_pool) > 0:
        distractors = random.choices(distractor_pool, k=3) 
    else:
        distractors = [data_item['object'] + "_1", data_item['object'] + "_2", data_item['object'] + "_3"]

    choice = [data_item['object']] + distractors
    random.shuffle(choice)
    answer_index = choice.index(data_item['object'])

    question = template['question'].replace("A", data_item['concept']).replace("B", data_item['object'])

    new_data = OrderedDict()
    for key, value in data_item.items():
        new_data[key] = value
        if key == "generated_text":
            new_data["question"] = question
            new_data["choice"] = choice
            new_data["answer"] = answer_index

    return new_data


def process_jsonl(input_file, output_file):
    data = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Skipping invalid line: {line}")

    relation_dict = {}
    for d in data:
        relation_dict.setdefault(d['relation'], []).append(d)

    output_data = []
    for d in data:
        same_relation = relation_dict.get(d['relation'], [])
        d_with_qa = generate_qa(d, same_relation, relation_templates)
        output_data.append(d_with_qa)

    with open(output_file, "w", encoding="utf-8") as f:
        for item in output_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Processed {len(data)} items, saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="conceptnet relation QA JSONL")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file path")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL file path")

    args = parser.parse_args()
    
    process_jsonl(args.input, args.output)
