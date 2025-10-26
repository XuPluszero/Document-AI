import json
import argparse
from typing import Any, Dict, List, Optional
from copy import deepcopy
import pandas as pd


def maybe_clean_prediction_of_empty(
    prediction: Any, ground_truth: Any
) -> Any:
    if not isinstance(prediction, dict) or not isinstance(ground_truth, dict):
        return prediction

    clean_prediction = deepcopy(prediction)
    for key in prediction:
        prediction_is_empty = prediction[key] == None or prediction[key] == ""
        ground_truth_is_empty = key not in ground_truth
        if prediction_is_empty and ground_truth_is_empty:
            clean_prediction.pop(key)

    return clean_prediction


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run document AI benchmark evaluation")
    parser.add_argument(
        "--model-generation-path",
        type=str,
        help="Path to the model generation JSON file"
    )
    
    args = parser.parse_args()
    

    with open(args.model_generation_path) as f:
        results = json.load(f)
    evaluation_results = []
    num_api_error = 0
    num_extraction_error = 0
    num_is_correct = 0
    for each_result in results:
        is_correct = False
        wrong_prediction_type = None
        if each_result['response'] is None:
            wrong_prediction_type = "API error"
            num_api_error += 1
        elif each_result['result'] is None or 'extraction' not in each_result['result']:
            wrong_prediction_type = "Extraction error"
            num_extraction_error += 1
        else:
            prediction = each_result['result']['extraction']
            ground_truth = each_result['ground_truth']
            clean_prediction = maybe_clean_prediction_of_empty(
                prediction, ground_truth
            )
            if ground_truth is None and clean_prediction is None:
                is_correct = True
            elif ground_truth is None and clean_prediction is not None:
                wrong_prediction_type = "False positive"
            elif ground_truth is not None and clean_prediction is None:
                wrong_prediction_type = "False negative"
            elif ground_truth != clean_prediction:
                wrong_prediction_type = "Incorrect value"
            else:
                is_correct = True
        if is_correct:
            num_is_correct += 1
        evaluation_results.append({
            "doc_name": each_result['doc_name'],
            "line_item_name": each_result['line_item_name'],
            "response": each_result['response'],
            "ground_truth": each_result['ground_truth'],
            "is_correct": is_correct,
            "wrong_prediction_type": wrong_prediction_type,
        })
    print(f"Num API error: {num_api_error}, percentage: {num_api_error / len(evaluation_results)}")
    print(f"Num extraction error: {num_extraction_error}, percentage: {num_extraction_error / len(evaluation_results)}")
    print(f"Num is correct: {num_is_correct}, percentage: {num_is_correct / len(evaluation_results)}")
    with open(f"{args.model_generation_path.replace('.json', '_eval.json')}", 'w') as f:
        json.dump(evaluation_results, f, indent=4)