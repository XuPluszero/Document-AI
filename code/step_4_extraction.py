import json
import argparse
from copy import deepcopy
import json
import openai
from joblib import Parallel, delayed
from tqdm import tqdm
import argparse


EXTRACTION_INSTRUCTION = """
**POLICY DOCUMENT**
{document_sections}

**POLICY METADATA**
{policy_metadata}

**LINE ITEM DEFINITION**
{line_item_detail}

Your objective is to extract the line item from the policy document. You must follow the guidelines below:

1. Follow exactly the extraction instructions defined in the `Line item instruction` field within **LINE ITEM DEFINITION**
2. Go through each section in the **POLICY DOCUMENT** throughly and check whether it contains definitions or phrases related to the extraction item, list ALL the evidence that supports your reasoning for the extraction task.
3. Refer to **POLICY METADATA** for policy-level information and the coverage limits.
4. (IMPORTANT) In insurance policies, endorsement can modify earlier definitions, if there exists endorsement that modifies the parameters of this extraction item, list ALL the evidence and your reasoning how this endorsement affects the extraction item.
5. (IMPORTANT) There is NO DEFAULT VALUE for the line item parameters. If you cannot find the values for the parameters, you must leave the parameters as null and DO NOT ASSUME ANY VALUES.

## OUTPUT
Once you have all the information, call `extract` tool to output result.
""".strip(
    "\n "
)


def process_metadata(extractor_results):
    policy_conditions = extractor_results['policy_conditions']
    sub_limits = extractor_results['sub_limits']
    interesting_keys = [
        'aggregate_limit_of_liability',
        'premium',
        'retention',
        'waiting_period',
        'indemnity_period',
    ]
    policy_level_info = {}
    for key in policy_conditions:
        if key in interesting_keys and policy_conditions[key] != 0 and policy_conditions[key] != '':
            policy_level_info[key] = policy_conditions[key]
    coverage_limits = []
    for each_sub_limit in sub_limits:
        to_record = {}
        for key in each_sub_limit:
            if each_sub_limit[key] is not None and each_sub_limit[key] != '' and each_sub_limit[key] != 0:
                to_record[key] = each_sub_limit[key]
        coverage_limits.append(to_record)
    return {
        "policy_level_info": policy_level_info,
        "coverage_limits": coverage_limits,
    }


def process_item(item_info):
    """Process a single schema item for a document."""
    client = openai.OpenAI()
    try:
        response = client.chat.completions.create(**item_info['create_params'])
        return response.model_dump()
    except Exception as e:
        print(f"API Error: {item_info['doc_name']} - {item_info['item_name']} - {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Create benchmark dataset for Osprey document AI")
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
        "--reasoning-model", 
        choices=['YES', 'NO'],
        default='NO',
        required=False,
        help="Use reasoning model: YES or NO"
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=16,
        required=False,
        help="Number of jobs"
    )
    
    args = parser.parse_args()
    
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
    
    with open('raw_data/extraction_instructions.json', 'r') as f:
        extraction_instructions = json.load(f)
    line_items = [each_item['Line item name'] for each_item in extraction_instructions]
    if args.test_run:
        line_items = line_items[:1]
    instruction_dct = {each_item['Line item name']: each_item for each_item in extraction_instructions}
    if args.test_run:
        print(f"This is a dry run. Only processing the first document and line item. {docs[0]} {line_items[0]}")

    full_doc_samples = []
    retrieved_doc_samples = []
    for doc_name in docs:
        with open(f"raw_data/ground_truths/{doc_name}.json", "r") as f:
            gt = json.load(f)
        with open(f"raw_data/outputs/{doc_name}.json", "r") as f:
            extraction_input = json.load(f)
        relevant_sections_per_line_item = {}
        for each_line_item_result in extraction_input['results']:
            line_item_name = each_line_item_result['retrieval_result']['line_item_name']
            relevant_sections_per_line_item[line_item_name] = each_line_item_result['retrieval_result']['relevant_sections']

        policy_metadata = process_metadata(extraction_input)

        for line_item_name in line_items:
            ground_truth = gt['synthesizer_result'][line_item_name]
            ref_sections_full_doc = []
            ref_sections_retrieved = []
            for each_section in extraction_input['chunker_result']['document_sections']:
                content = f"==== SECTION ID: {each_section['id']} ====\n{each_section['title']}\n{each_section['text']}\n==== SECTION {each_section['id']} END ===="
                ref_sections_full_doc.append(content)
                if each_section['id'] in relevant_sections_per_line_item[line_item_name]:
                    ref_sections_retrieved.append(content)
            assert len(ref_sections_full_doc) > 0 and len(ref_sections_retrieved) > 0
            ref_content_full_doc = "\n\n".join(ref_sections_full_doc)
            ref_content_retrieved = "\n\n".join(ref_sections_retrieved)
            line_item_detail = {
                "Line item instruction": instruction_dct[line_item_name]['Line item instruction'],
                "Line item schema": instruction_dct[line_item_name]['Line item schema'],
            }
            prompt_full_doc = EXTRACTION_INSTRUCTION.format(
                document_sections=ref_content_full_doc,
                policy_metadata=json.dumps(policy_metadata, indent=4),
                line_item_detail=json.dumps(line_item_detail, indent=4)
            )
            prompt_retrieved = EXTRACTION_INSTRUCTION.format(
                document_sections=ref_content_retrieved,
                policy_metadata=json.dumps(policy_metadata, indent=4),
                line_item_detail=json.dumps(line_item_detail, indent=4)
            )
            this_extraction_obj = deepcopy(instruction_dct[line_item_name]['Line item schema'])
            this_extraction_obj["description"] = "The extracted object for this line item. If the conclusion in your thinking process is `No evidence found`, output null."
            
            # Build tool properties based on reasoning flags
            tool_properties = {}
            required_fields = []
            if args.reasoning_model == 'NO':
                tool_properties["think"] = {
                    "type": "string",
                    "description": "Output your detailed thinking process for the extraction task.",
                }
                required_fields.append("think")
            tool_properties["extraction"] = this_extraction_obj
            required_fields.append("extraction")
            this_tool = {
                "type": "function",
                "function": {
                    "name": "extract",
                    "parameters": {
                        "type": "object",
                        "properties": tool_properties,
                        "required": required_fields,
                        "additionalProperties": False,
                    },
                },
            }
            temperature = 0.0 if args.reasoning_model == 'NO' else 0.6
            max_tokens = 10000 if args.reasoning_model == 'NO' else 32000
            full_doc_samples.append({
                "doc_name": doc_name,
                "line_item_name": line_item_name,
                "ground_truth": ground_truth,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt_full_doc,
                    },
                ],
                "temperature": temperature,
                "tools": [this_tool],
                "max_tokens": max_tokens,
            })
            retrieved_doc_samples.append({
                "doc_name": doc_name,
                "line_item_name": line_item_name,
                "ground_truth": ground_truth,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt_retrieved,
                    },
                ],
                "temperature": temperature,
                "tools": [this_tool],
                "max_tokens": max_tokens,
            })

    print(f"Number of doc samples: {len(full_doc_samples)}")

    # with open(f"processed_data/step_4_extraction_prompt_full_doc_reasoning_model_{args.reasoning_model}_tryrun_{args.test_run}.json", "w") as f:
    #     json.dump(full_doc_samples, f, indent=4)
    # with open(f"processed_data/step_4_extraction_prompt_retrieved_doc_reasoning_model_{args.reasoning_model}_tryrun_{args.test_run}.json", "w") as f:
    #     json.dump(retrieved_doc_samples, f, indent=4)


    print(f"Running model generation...")
    extraction_results = {doc_name: {} for doc_name in docs}
    processing_units = []
    for each_sample in retrieved_doc_samples:
        create_params = {
            "messages": each_sample['messages'],
            "tools": each_sample['tools'],
            "max_tokens": each_sample['max_tokens'],
            "temperature": each_sample['temperature'],
            "model": args.model_name,
        }
        each_unit = {
            "doc_name": each_sample['doc_name'],
            "line_item_name": each_sample['line_item_name'],
            'create_params': create_params,
        }
        if 'ground_truth' in each_sample:
            each_unit['ground_truth'] = each_sample['ground_truth']
        processing_units.append(each_unit)

    results = Parallel(n_jobs=args.n_jobs)(
        delayed(process_item)(item_info)
        for item_info in tqdm(processing_units)
    )
    print(f"\n\nGathering results...")
    for i, each_unit in enumerate(processing_units):
        each_unit['response'] = results[i]
        try:
            each_unit['result'] = json.loads(results[i]['choices'][0]['message']['tool_calls'][0]['function']['arguments'])
            if 'think' in each_unit['result']:
                each_unit['reasoning'] = each_unit['result']['think']
            else:
                each_unit['reasoning'] = None
        except Exception as e:
            each_unit['result'] = None
            each_unit['reasoning'] = None
            print(f"Extraction Error: {each_unit['doc_name']} - {each_unit['line_item_name']} - {e}")
        extraction_results[each_unit['doc_name']][each_unit['line_item_name']] = {
            "reasoning": each_unit['reasoning'],
            "result": each_unit['result'],
        }

    with open(f"processed_data/step_4_extraction_result_{args.model_name}.json", "w") as f:
        json.dump(extraction_results, f, indent=4)
    with open(f"processed_data/step_4_extraction_log_{args.model_name}.json", "w") as f:
        json.dump(processing_units, f, indent=4)



if __name__ == "__main__":
    main()