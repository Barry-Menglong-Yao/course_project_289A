import json
import random
import argparse
from collections import OrderedDict
from templates_tf import tf_relation_templates

def generate_tf_qa(data_item, templates):
    relation = data_item.get("relation")
    concept_subject = data_item.get("subject")
    concept_object = data_item.get("object")

    if relation not in templates:
        return None

    template = random.choice(templates[relation])
    question = template["question"].replace("A", concept_subject).replace("B", concept_object)
    answer = template["answer"]

    return question, answer

def process_jsonl(input_file, output_file):
    data_out = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data_item = json.loads(line)
            question_answer = generate_tf_qa(data_item, tf_relation_templates)

            if question_answer:
                question, answer = question_answer

                new_item = OrderedDict()
                for key in data_item:
                    new_item[key] = data_item[key]

                    if key == "generated_text":
                        new_item["question"] = question
                        new_item["answer"] = answer

                data_out.append(new_item)
            else:
                data_out.append(OrderedDict(data_item))

    with open(output_file, "w", encoding="utf-8") as f_out:
        for item in data_out:
            f_out.write(json.dumps(item, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file path")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL file path")
    args = parser.parse_args()

    process_jsonl(args.input, args.output)
