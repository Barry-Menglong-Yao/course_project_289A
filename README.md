
---



## 📖Table of Contents


- 🔧[Installation](#installation)
- 🗃️[Dataset-Construction](#%EF%B8%8Fdataset-construction)
- ✏️[Training](#%EF%B8%8Ftraining)

---


## 🔧Installation

```bash
git clone https://github.com/zjunlp/DynamicKnowledgeCircuits
cd DynamicKnowledgeCircuits
pip install -r requirements.txt
```

## 🗃️Dataset-Construction

**Step1:** Generate the knowledge entities with different types and frequencies.

```bash
python data/datapoint_generation.py
```

**Step2:** Convert the knowledge entities into biography segments.

```bash
python data/text_data_generation.py
```

**Step3:** Generate the test set for the evaluation of models and knowledge circuits.

```bash
python data/query_data_generation.py
python data/circuits_data_generation.py
```

## Specific dataset construction ##

1. Generate a specialized training dataset where one selected high-level knowledge category is always trained before another, while all other categories remain in their original positions.

```bash 
python generate_knowledge_interference_train.py
```

2. Generate a test dataset that contains only one selected high-level knowledge category, and write those filtered instances to a new file.

```bash 
python generate_knowledge_interference_test.py
```

3. Generate a training dataset by filtering out all non-universal or overly specific statements that may introduce noise or harm the model’s generalization.

```bash 
python filter.py
```
## QA data construction ##

1、Generate the multi-choice QA format using predefined templates

```bash
python Multichoice-data-construct.py --input {A_path_file}  --output {B_path_file}
```

2、Generate the true-false QA format using predefined templates

```bash
python TrueFalse-data-construct.py --input {A_path_file}  --output {B_path_file}
```



## ✏️Training

Run the continual pre-training process with the following command:

```bash
bash scripts/$model/train.sh
```
> Note: Remember to change `$model` in the script to the name model you want to train, such as `gpt2`, `gpt2-medium`, etc.

Example script for training GPT-2 small model:

```bash
model=gpt2
model_name_or_path=/mnt/8t/oyx/PLMs/${model}  # Change this to the path of the model
train_file=data/entities_50000/train.jsonl
validation_file=data/entities_50000/validation.jsonl
CUDA_VISIBLE_DEVICES=0,1 python train.py \
    --model $model \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name $model_name_or_path \
    --train_file $train_file \
    --validation_file $validation_file \
    --load_data_from_cache False \
    --block_size 1024 \
    --output_dir outputs/train/$model/$(date +"%Y-%m-%d-%H-%M-%S")/checkpoints \
    --do_train \
    --do_eval \
    --eval_strategy steps \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-3 \
    --weight_decay 0.1 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --adam_epsilon 1e-6 \
    --num_train_epochs 25 \
    --lr_scheduler_type constant \
    --logging_dir outputs/train/$model/$(date +"%Y-%m-%d-%H-%M-%S") \
    --logging_strategy steps \
    --logging_steps 50 \
    --save_strategy epoch \
    --report_to wandb
```


## 🔍Circuit-Discovery

Run the circuit discovery process with the following command:

```bash
bash scripts/$model/eap.sh
```
> Note: Remember to change `$model` in the script to the name model you want to train, such as `gpt2`, `gpt2-medium`, etc.

Example script for discovering knowledge circuits in GPT-2 small model:

```bash
model=gpt2
directory_path=outputs/train/gpt2/2024-12-19-22-27-33/checkpoints  # Change this to the path of the directory where the checkpoints are saved
circuit_n=300

for task in "city" "company" "major"; do
    data_file=data/entities_50000/circuit_${circuit_n}/${model}/${task}.jsonl
    for dirname in "$directory_path"/checkpoint-*/; do
        model_path="$dirname"
        for type in "new" "revised"; do
            for frequency in "high" "medium" "low"; do
                CUDA_VISIBLE_DEVICES=0 python circuit_discovery.py \
                    --model $model \
                    --model_path $model_path \
                    --task $task \
                    --data_file $data_file \
                    --type $type \
                    --frequency $frequency \
                    --batch_size 64 \
                    --method "EAP-IG"
            done
        done
    done
done
```

## 🤔Circuit-Evaluation

Run the circuit evaluation process with the following command:

```bash
bash scripts/$model/circuit_eval.sh
```
> Note: Remember to change `$model` in the script to the name model you want to train, such as `gpt2`, `gpt2-medium`, etc.

Example script for evaluate knowledge circuits in GPT-2 small model:

```bash
model=gpt2
directory_path=outputs/train/gpt2/2024-12-19-22-27-33/checkpoints  # Change this to the path of the directory where the checkpoints are saved
circuit_n=300

test_data_file=data/entities_50000/test.jsonl

for task in "city" "company" "major"; do
    eval_data_file=data/entities_50000/circuit_${circuit_n}/${model}/${task}.jsonl
    for dirname in "$directory_path"/checkpoint-*/; do
        model_path="$dirname"
        for source_type in "new" "revised"; do
            for source_frequency in "high" "medium" "low"; do
                target_type=$source_type
                for target_frequency in "high" "medium" "low"; do
                    CUDA_VISIBLE_DEVICES=0 python circuit_eval.py \
                        --model $model \
                        --model_path $model_path \
                        --task $task \
                        --eval_data_file $eval_data_file \
                        --test_data_file $test_data_file \
                        --source_type $source_type \
                        --source_frequency $source_frequency \
                        --target_type $target_type \
                        --target_frequency $target_frequency \
                        --batch_size 64 \
                        --method "EAP-IG" \
                        --topn 8000   # Change this to the number of edges that retained in the circuit
                done
            done
        done
    done
done
```



For RQ1: How do LLMs’ internal mechanisms correlate with the difficulty of acquiring different concepts?
```bash
bash scripts/gpt2/train_concept_v5_one_knowledge_per_relation_mcqa.sh
```

For RQ2: How do LLMs’ internal mechanisms correlate with the propensity to forget learned concepts?
```bash
bash scripts/gpt2/train_concept_v5_one_knowledge_per_relation_mcqa.sh
```

For RQ3: How does training on multiple concepts induce interference or synergy across concepts?
```bash
bash scripts/gpt2/train_concept_interference_groups.sh
```

For RQ4: How does training on multiple knowledge induce interference or synergy across knowledge? Run the following command to obtain the results. You can change the data and output directories based on your own file path.

```bash
bash scripts/gpt2/train_concept_v4_multi_knowledge.sh
```

After obtaining the results, you can run visualization to see the impact of how training on multiple knowledge induces interference or synergy across knowledge.

```bash
python visualize_result.py
```



