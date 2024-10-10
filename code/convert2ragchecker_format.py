import os
import json
import codecs
import argparse

from utils import process_response

def load_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return [json.loads(line.strip()) for line in file]

def save_as_jsonl(data, output_file):
    with open(output_file, 'w', encoding='utf-8') as file:
        for entry in data:
            json.dump(entry, file, ensure_ascii=False)
            file.write('\n')

def convert_data(prediction_file, output_file):
    with codecs.open(prediction_file, 'r', 'utf-8') as fr:
        predictions = json.load(fr)

    results = []
    for prediction in predictions:
        query_id = prediction['meta_data']["qid"]
        query = prediction["query"]
        gt_answer = prediction['meta_data']["answer"]
        retrieved_context = [dict(doc_id=str(i), text=p) for i, p in enumerate(prediction['passages'])]
        response = process_response(prediction['pred'])

        result = {
            "query_id": query_id,
            "query": query,
            "gt_answer": gt_answer,
            "retrieved_context": retrieved_context,
            "response": response
        }

        results.append(result)
    
    output_dir = os.path.split(output_file)[0]
    os.makedirs(output_dir, exist_ok=True)
    with codecs.open(output_file, 'w', 'utf-8') as fw:
        json.dump(dict(results=results), fw, ensure_ascii=False)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Model related options
    parser.add_argument('--prediction_file', type=str, help="File of prediction data being converted.")
    parser.add_argument('--output_file', type=str, help="File of converted data to be saved.")
    args = parser.parse_args()

    convert_data(prediction_file=args.prediction_file, output_file=args.output_file)
