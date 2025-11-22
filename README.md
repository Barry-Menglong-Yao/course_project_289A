
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

##QA data construction##

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

For RQ4: How does training on multiple knowledge induce interference or synergy across knowledge? Run the following command to obtain the results. You can change the data and output directories based on your own file path.

```bash
bash train_concept_v4_multi_knowledge.sh
```

After obtaining the results, you can run visualization to see the impact of how training on multiple knowledge induces interference or synergy across knowledge.

```bash
python visualize_result.py
```



