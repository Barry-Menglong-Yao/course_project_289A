import json
from vllm import LLM, SamplingParams

# Define general (universal) relations
GENERAL_RELATIONS = {
    "IsA", "HasA", "HasProperty", "CapableOf", "UsedFor",
    "PartOf", "MadeOf", "Desires", "CreatedBy","AtLocation",
    "CausesDesire","Synonym","Antonym"
}

# Initialize model once (outside any subprocess)
llm = LLM(model="Qwen/Qwen3-8B")
sampling_params = SamplingParams(temperature=0.7, top_p=0.8, top_k=20, max_tokens=10)

def classify_statement(data):
    statement = data.get("generated_text", "")
    relation = data.get("relation", "")
    concept = data.get("concept")
    conceptnet_id = data.get("conceptnet_id")

    if concept and conceptnet_id and statement:
        # replace only the first occurrence
        statement = statement.replace(concept, conceptnet_id, 1)

    if not statement:
        return None

    # If relation is in general set, automatically classify as General
    if relation in GENERAL_RELATIONS:
        return "General", json.dumps(data, ensure_ascii=False) + "\n"

    # Only run LLM for relations NOT in the general set
    prompt = (
        "Classify the following statement as *General* (universal property) "
        "or *Specific* (non-universal detail). Answer only with General or Specific.\n\n"
        """General/Universal traits (biological features, common abilities, essential properties, functional properties, structural properties):
        - "Dogs have four legs" - biological feature
        - "Birds can fly" - common ability
        - "Water is wet" - essential property
        - "Knives are used for cutting" - functional property
        - "Cars have wheels" - structural property
        - "Gorilla fits the category of ape"
        - "dynamite has the potential to explode"
        - "peppermint is sometimes termed red gum" - even though it has Specific term red, it is still classified as universal because this content is try to define synonym.
        - "snow is white" - even though it has Specific term white, it is still classified as universal because snow has only one color which is white.

        Specific/Non-universal traits (colors, specific sizes, specific times, personal attributes):
        - "The dogs are white" - color varies, can be white, can be brown,etc.
        - "The building is 100 feet tall" - specific size
        - "The car was made in 2020" - specific time"""

        f"Statement: \"{statement}\"\n"
        "Classification:"
    )

    outputs = llm.generate([prompt], sampling_params)
    result_text = outputs[0].outputs[0].text.strip().lower()

    if "general" in result_text:
        return "General", json.dumps(data, ensure_ascii=False) + "\n"
    else:
        data["predicted_category"] = "Specific"
        return "Specific", json.dumps(data, ensure_ascii=False) + "\n"

if __name__ == "__main__":
    INPUT_FILE = "t6.4_concept_separate_knowledge_test.jsonl"
    OUT_GENERAL = "t6.4_concept_separate_knowledge_general_qwen.jsonl"
    OUT_FILTERED_LOG = "t6.4_concept_separate_knowledge_filtered_log_qwen.jsonl"

    general_count = 0
    specific_count = 0 

    with open(INPUT_FILE, "r", encoding="utf-8") as infile:
        lines = infile.readlines()

    with open(OUT_GENERAL, "w", encoding="utf-8") as out_gen, \
         open(OUT_FILTERED_LOG, "w", encoding="utf-8") as out_log:

        for line in lines:
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            result = classify_statement(data)
            if not result:
                continue

            category, out_line = result
            if category == "General":
                general_count += 1
                out_gen.write(out_line)
            else:
                specific_count += 1
                out_log.write(out_line)

    print("Processing completed!")
    print(f"General statements saved to: {OUT_GENERAL}")
    print(f"Specific statements saved to: {OUT_FILTERED_LOG}")
    print()
    print(f"Total General:  {general_count}")
    print(f"Total Specific: {specific_count}")