import json
import re

# Step 1: Relation-to-type mapping
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
    # "CausesDesire": "Desire",
    # "Desires": "Desire",
    "Causes": "Causality & Event Structure",
    "MotivatedByGoal": "Causality & Event Structure",
    "ObstructedBy": "Causality & Event Structure",
    "HasPrerequisite": "Causality & Event Structure",
    "HasSubevent": "Causality & Event Structure",
    "HasFirstSubevent": "Causality & Event Structure",
    "HasLastSubevent": "Causality & Event Structure",
    "CreatedBy": "Causality & Event Structure",
    # "DerivedFrom": "Lexical/Etymological",
    # "EtymologicallyDerivedFrom": "Lexical/Etymological",
    # "EtymologicallyRelatedTo": "Lexical/Etymological",
}

# Step 2: Automatically extract all unique relation types from the dictionary
unique_types = sorted(set(relation_2_relation_type_dict.values()))

print("Available relation types:")
for i, rel_type in enumerate(unique_types):
    print(f"  {i}: {rel_type}")

# Step 3: Generate all permutations of TYPE1 and TYPE2 (order matters: (A,B) != (B,A))
from itertools import permutations

type_combinations = list(permutations(unique_types, 2))
print(f"\nFound {len(type_combinations)} type pairs to process (order matters)")

# Step 4: Input file path
input_path = "data/concept/v3/t6.4_concept_separate_knowledge_train.jsonl"

# Step 5: Dynamically generate output filename based on type initials
def make_abbreviation(text):
    return ''.join(word[0].upper() for word in re.findall(r'\b\w+', text))

# Step 6: Process each combination
for TYPE1, TYPE2 in type_combinations:
    print(f"\n{'='*60}")
    print(f"Processing: TYPE1={TYPE1}, TYPE2={TYPE2}")
    print(f"{'='*60}")
    
    # Categorization logic
    def categorize_relation(relation):
        group = relation_2_relation_type_dict.get(relation, "Other")
        if group == TYPE1:
            return "TYPE1"
        elif group == TYPE2:
            return "TYPE2"
        else:
            return "Other"
    
    # Read and categorize all lines
    lines_with_categories = []
    type1_number = 0
    type2_number = 0
    with open(input_path, 'r', encoding='utf-8') as infile:
        for line in infile:
            try:
                record = json.loads(line)
                relation_type = record.get("relation")
                category = categorize_relation(relation_type)
                lines_with_categories.append((category, line.strip()))
                if category == "TYPE1":
                    type1_number += 1
                elif category == "TYPE2":
                    type2_number += 1
            except json.JSONDecodeError:
                lines_with_categories.append(("Other", line.strip()))
    if type1_number <100 or type2_number <100:
        print(f"Skipping {TYPE1} and {TYPE2} because one of them has no lines")
        continue
    # Find first TYPE2 index and collect TYPE1 lines after it
    type1_lines_after_type2 = []
    lines_before_first_type2 = []
    lines_not_type1_from_first_type2 = []
    lines_type1_after_first_type2 = []
    type2_first_index = None
    
    for i, (category, line) in enumerate(lines_with_categories):
        if category == "TYPE2":
            if type2_first_index is None:
                type2_first_index = i  # Record the first TYPE2 index
            lines_not_type1_from_first_type2.append(line)
        elif category != "TYPE1":
            if type2_first_index is None:
                lines_before_first_type2.append(line)
            else:
                lines_not_type1_from_first_type2.append(line)
        elif category == "TYPE1":
            if type2_first_index is not None:
                lines_type1_after_first_type2.append(line)
            else:
                lines_before_first_type2.append(line)
    
    # Reorder lines - move TYPE1 lines (before first TYPE2) to before first TYPE2
    output_lines = []
    
    for one_line in lines_before_first_type2:
        output_lines.append(one_line + '\n')
    for one_line in lines_type1_after_first_type2:
        output_lines.append(one_line + '\n')
    for one_line in lines_not_type1_from_first_type2:
        output_lines.append(one_line + '\n')
    
    # Generate output filename
    type1_abbr = make_abbreviation(TYPE1)
    type2_abbr = make_abbreviation(TYPE2)
    output_path = f"data/concept/v3/rq3/t6.4_concept_separate_knowledge_train_reorder_{type1_abbr}_{type2_abbr}.jsonl"
    
    # Write output
    with open(output_path, 'w', encoding='utf-8') as outfile:
        outfile.writelines(output_lines)
    
    print(f"✓ Output written to: {output_path}")

print(f"\n{'='*60}")
print(f"All {len(type_combinations)} combinations processed!")
print(f"{'='*60}")
