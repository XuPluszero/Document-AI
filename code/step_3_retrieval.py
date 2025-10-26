import json
import openai
import argparse
from transformers import AutoTokenizer
from joblib import Parallel, delayed
from tqdm import tqdm


TOKENIZER = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

tool_def = {
    "type": "function",
    "function": {
        "name": "output",
        "parameters": {
            "type": "object",
            "properties": {
                "think": {
                    "type": "string",
                    "description": "Output your detailed thinking."
                },
                "relevant_sections": {
                    "type": "array",
                    "description": "List of section IDs that found evidence related to the targets.",
                    "items": {
                        "type": "string",
                        "description": "Section ID"
                    }
                },
            },
            "required": ["think", "relevant_sections"],
            "additionalProperties": False,
        }
    }
}

RECALL_INSTRUCTION = """
**DOCUMENT SECTION LIST**
{document_section_list}

**LINE ITEM INSTRUCTION**
{line_item_detail}

Your objective is to determine whether each section in **DOCUMENT SECTION LIST** contains the information that **LINE ITEM INSTRUCTION** is looking for. You MUST follow the guidelines below: 
## MANDATORY THINKING PROCESS
1. Read through **LINE ITEM INSTRUCTION** and understand the targets to look for in the document section.
2. Go through each section throughly and check whether it contains info related to the targets. For EVERY section, write down the evidence and reasoning why it is related in the format of `- [Section ID]: evidence: [Evidence in this section], reasoning: [Reasoning for this section].\n`.

## OUTPUT
You MUST call `output` tool to output result.
""".strip("\n ")


def get_client():
    client = openai.OpenAI()
    return client


def process_item(item_info):
    """Process a single schema item for a document."""
    # print(f"Processing {item_info['doc_name']} - {item_info['item_name']} - {item_info['section_group_idx']}...")
    try:
        response = get_client().chat.completions.create(**item_info['create_params'])
        return response.model_dump()
    except Exception as e:
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run recall evaluation on policy documents')
    parser.add_argument(
        "--test-run",
        type=bool,
        default=True,
        required=False,
        help="Test run"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="gpt-4.1",
        required=False,
        help="Model name"
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=16,
        required=False,
        help="Number of jobs"
    )
    
    args = parser.parse_args()
    
    TEMPERATURE = 0.0
    MAX_TOKENS = 5000
    N_JOBS = 30
    SECTION_BATCH_SIZE = 10
    SECTION_MAX_TOKENS = 10000

    docs = [
        "adventis",
        "ancora_heart",
        "andrian",
        "anomali",
        "at_bay",
        "bitgo",
        "corium",
        "gardner",
        "jfrog",
        "park_place",
        "people_ai",
        "sprout",
        "standard_biotools",
        "sylabs",
    ]
    if args.test_run:
        docs = docs[:1]
    with open("raw_data/retrieval_instructions.json", "r") as f:
        line_item_descs = json.load(f)
    if args.test_run:
        line_item_descs = line_item_descs[:1]
    line_item_descs_dct = {d['Line item name']: d for d in line_item_descs}

    processing_units = []
    all_sections = {}
    for doc_name in docs:
        with open(f"raw_data/outputs/{doc_name}.json", "r") as f:
            chunker_result = json.load(f)['chunker_result']
        sections = chunker_result['document_sections']
        all_sections[doc_name] = {s['id']: s for s in sections}
        grouped_sections = []
        current_section_index = 0
        last_local_group = []
        last_local_group_token_count = 0
        while current_section_index < len(sections):
            last_local_group.append(sections[current_section_index])
            last_local_group_token_count += len(TOKENIZER.encode(json.dumps(sections[current_section_index], indent=4)))
            if last_local_group_token_count > SECTION_MAX_TOKENS or len(last_local_group) >= SECTION_BATCH_SIZE:
                grouped_sections.append(last_local_group)
                last_local_group = []
                last_local_group_token_count = 0
            current_section_index += 1
        if len(last_local_group) > 0:
            grouped_sections.append(last_local_group)
        print(f"{doc_name} Grouped {len(grouped_sections)} sections")

        for item_name in line_item_descs_dct:
            item_instruction = line_item_descs_dct[item_name]['Line item instruction']
            for group_idx, local_sections in enumerate(grouped_sections):
                ref_sections = []
                for each_section in local_sections:
                    ref_sections.append(f"==== SECTION ID: {each_section['id']} ====\n{each_section['title']}\n{each_section['text']}\n==== SECTION {each_section['id']} END ====")
                document_section_list = "\n\n".join(ref_sections)
                prompt = RECALL_INSTRUCTION.format(
                    document_section_list=document_section_list,
                    line_item_detail=item_instruction,
                )

                create_params = {
                    "model": args.model_name,
                    "messages": [{'role': 'user', 'content': prompt}],
                    "temperature": TEMPERATURE,
                    "tools": [tool_def],
                    "max_tokens": MAX_TOKENS,
                }
                each_unit = {
                    "doc_name": doc_name,
                    "item_name": item_name,
                    "section_group_idx": group_idx,
                    "section_ids": [s['id'] for s in local_sections],
                    'create_params': create_params,
                }
                processing_units.append(each_unit)

    results = Parallel(n_jobs=N_JOBS)(
        delayed(process_item)(item_info)
        for item_info in tqdm(processing_units)
    )
    print(f"\n\nGathering results...")
    for i, each_unit in enumerate(processing_units):
        each_unit['response'] = results[i]
        try:
            each_unit['result'] = json.loads(results[i]['choices'][0]['message']['tool_calls'][0]['function']['arguments'])
            each_unit['reasoning'] = each_unit['result']['think']
        except Exception as e:
            each_unit['result'] = None
            each_unit['reasoning'] = None
            print(f"Error: {each_unit['doc_name']} - {each_unit['item_name']} - {each_unit['section_group_idx']} - {e}")
    
    recall_results = {doc_name: {} for doc_name in docs}
    for each_unit in processing_units:
        if each_unit['item_name'] not in recall_results[each_unit['doc_name']]:
            recall_results[each_unit['doc_name']][each_unit['item_name']] = {'result_lst': []}
        recall_results[each_unit['doc_name']][each_unit['item_name']]['result_lst'].append(each_unit)
    # Aggregate results
    for doc_name, items_per_doc in recall_results.items():
        for item_name, item_info in items_per_doc.items():
            relevant_section_ids = []
            reasoning_lst = []
            for each_unit in item_info['result_lst']:
                try:
                    assert isinstance(each_unit['result']['relevant_sections'], list)
                    relevant_section_ids.extend(each_unit['result']['relevant_sections'])
                    reasoning_lst.append(each_unit['reasoning'])
                except Exception as e:
                    print(f"Aggregate results Error: {each_unit['doc_name']} - {each_unit['item_name']} - {each_unit['section_group_idx']} - {e}. Adding all section ids: {each_unit['section_ids']}")
                    relevant_section_ids.extend(each_unit['section_ids'])
            item_info['reasoning'] = "\n\n".join(reasoning_lst)
            item_info['relevant_sections'] = sorted(list(set(relevant_section_ids)))
            item_info.pop('result_lst')
            recall_sections = []
            for section_id in item_info['relevant_sections']:
                if section_id in all_sections[doc_name]:
                    recall_sections.append(all_sections[doc_name][section_id])
                else:
                    print(f"Section ID {section_id} not found in {doc_name}")
            token_count = len(TOKENIZER.encode(json.dumps(recall_sections, indent=4)))
            item_info['token_count'] = token_count
    with open(f"processed_data/step_3_retrieval_result_{args.model_name}.json", 'w') as f:
        json.dump(recall_results, f, indent=4)
    with open(f"processed_data/step_3_retrieval_log_{args.model_name}.json", "w") as f:
        json.dump(processing_units, f, indent=4)
