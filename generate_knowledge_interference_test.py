import json
import re

# === Relation → Category Mapping ===
relation_2_relation_type_dict = {
    "IsA": "Hyponym and Hypernym",
    "DefinedAs": "Hyponym and Hypernym",
    "FormOf": "Hyponym and Hypernym",
    "InstanceOf": "Hyponym and Hypernym",
    "Synonym": "Synonym and Antonym",
    "SimilarTo": "Synonym and Antonym",
    "Antonym": "Synonym and Antonym",
    "DistinctFrom": "Synonym and Antonym",
    "PartOf": "Meronym and Holonym",
    "HasA": "Meronym and Holonym",
    "MadeOf": "Meronym and Holonym",
    "HasProperty": "Property and Affordance",
    "UsedFor": "Property and Affordance",
    "CapableOf": "Property and Affordance",
    "ReceivesAction": "Property and Affordance",
    "AtLocation": "Spatial Relation",
    "LocatedNear": "Spatial Relation",
    "CausesDesire": "Desire",
    "Desires": "Desire",
    "Causes": "Causality & Event Structure",
    "MotivatedByGoal": "Causality & Event Structure",
    "ObstructedBy": "Causality & Event Structure",
    "HasPrerequisite": "Causality & Event Structure",
    "HasSubevent": "Causality & Event Structure",
    "HasFirstSubevent": "Causality & Event Structure",
    "HasLastSubevent": "Causality & Event Structure",
    "CreatedBy": "Causality & Event Structure",
    "DerivedFrom": "Lexical/Etymological",
    "EtymologicallyDerivedFrom": "Lexical/Etymological",
    "EtymologicallyRelatedTo": "Lexical/Etymological",
}

# === Choose one (set the other to None) ===
TARGET_CATEGORY = "Lexical/Etymological"   # e.g. "Property and Affordance"
TARGET_RELATION = None                            # e.g. "CapableOf"

# === Input file ===
input_path = "t6.4_concept_separate_knowledge_test.jsonl"

# === Generate abbreviation ===
def make_abbreviation(text):
    return ''.join(word[0].upper() for word in re.findall(r'\b\w+', text))

# === Determine filtering set and filename suffix ===
if TARGET_RELATION:
    relations_to_keep = {TARGET_RELATION}
    suffix = make_abbreviation(TARGET_RELATION)
elif TARGET_CATEGORY:
    relations_to_keep = {r for r, cat in relation_2_relation_type_dict.items() if cat == TARGET_CATEGORY}
    suffix = make_abbreviation(TARGET_CATEGORY)
else:
    raise ValueError("You must specify either TARGET_RELATION or TARGET_CATEGORY.")

# === Output filename ===
output_path = f"t6.4_concept_separate_knowledge_test_{suffix}.jsonl"

# === Filter and write ===
count = 0
total = 0

with open(input_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', encoding='utf-8') as outfile:
    for line in infile:
        total += 1
        try:
            obj = json.loads(line)
            if obj.get("relation") in relations_to_keep:
                outfile.write(line)
                count += 1
        except json.JSONDecodeError:
            continue

print(f"Kept {count} of {total} lines with relation(s): {sorted(relations_to_keep)}")
print(f"Output written to: {output_path}")
