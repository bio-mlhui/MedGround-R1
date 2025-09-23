from transformers import  AutoTokenizer, AutoProcessor
import torch
import json
from tqdm import tqdm
import re
import os
from pprint import pprint
import random
from transformers import AutoTokenizer, AutoModel
steps = 450
print("Steps: ", steps)
MODEL_PATH=f"/root/workspace/VLM-R1/src/open-r1-multimodal/src/open_r1/output/debug_med2/checkpoint-{steps}" 
OUTPUT_PATH="./logs/rec22_results_{DATASET}_qwen2_5vl_3b_instruct_r1_{STEPS}.json"
BSZ=32
DATA_ROOT = "/root/workspace/VLM-R1/root/datasets/MedRPG/data/MS_CXR"

# TEST_DATASETS = ['refcoco_val', 'refcocop_val', 'refcocog_val']
# IMAGE_ROOT = "/data/shz/dataset/coco"
TEST_DATASETS = ['cxr_test']
IMAGE_ROOT = "/root/workspace/VLM-R1/root/datasets/MedRPG/ln_data/MS_CXR"
from internvl.model.internvl_chat import InternVLChatModel
random.seed(42)

#We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
model = InternVLChatModel.from_pretrained(
        args.checkpoint, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16,
        load_in_8bit=args.load_in_8bit).eval()
# default processer
processor = AutoProcessor.from_pretrained(MODEL_PATH)

def extract_bbox_answer(content):
    # Try to find the bbox within <answer> tags, if can not find, return [0, 0, 0, 0]
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    bbox_pattern = r'\{.*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)]\s*.*\}'
    content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
    if content_answer_match:
        content_answer = content_answer_match.group(1).strip()
        bbox_match = re.search(bbox_pattern, content_answer, re.DOTALL)
        if bbox_match:
            bbox = [int(bbox_match.group(1)), int(bbox_match.group(2)), int(bbox_match.group(3)), int(bbox_match.group(4))]
            x1, y1, x2, y2 = bbox
            return bbox, False
    return [0, 0, 0, 0], False

def iou(box1, box2):
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2]-1, box2[2]-1)
    inter_y2 = min(box1[3]-1, box2[3]-1)
    if inter_x1 < inter_x2 and inter_y1 < inter_y2:
        inter = (inter_x2-inter_x1+1)*(inter_y2-inter_y1+1)
    else:
        inter = 0
    union = (box1[2]-box1[0])*(box1[3]-box1[1]) + (box2[2]-box2[0])*(box2[3]-box2[1]) - inter
    return float(inter)/union

sample_num = 500

for ds in TEST_DATASETS:
    print(f"Processing {ds}...")
    ds_path = os.path.join(DATA_ROOT, f"{ds}.json")
    data = json.load(open(ds_path, "r"))
    random.shuffle(data)
    QUESTION_TEMPLATE = "{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags. Output the final answer in JSON format."
    data = data[:sample_num]
    messages = []

    for x in data:
        image_path = os.path.join(IMAGE_ROOT, x['image'])
        message = [
            # {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {
            "role": "user",
            "content": [
                {
                    "type": "image", 
                    "image": f"file://{image_path}"
                },
                {
                    "type": "text",
                    "text": QUESTION_TEMPLATE.format(Question=x['problem'])
                }
            ]
        }]
        messages.append(message)

    all_outputs = []  # List to store all answers

    # Process data
    for i in tqdm(range(0, len(messages), BSZ)):
        batch_messages = messages[i:i + BSZ]
    
        # Preparation for inference
        text = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch_messages]
        
        image_inputs, video_inputs = process_vision_info(batch_messages)
        inputs = processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda:0")

        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, use_cache=True, max_new_tokens=256, do_sample=False)
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        batch_output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        all_outputs.extend(batch_output_text)
        # print(f"Processed batch {i//BSZ + 1}/{(len(messages) + BSZ - 1)//BSZ}")

    final_output = []
    correct_number = 0

    for input_example, model_output in zip(data, all_outputs):
        original_output = model_output
        ground_truth = input_example['solution']
        ground_truth_normalized = input_example['normalized_solution']
        model_answer, normalized = extract_bbox_answer(original_output)
        
        # Count correct answers
        correct = 0
        if model_answer is not None:
            if not normalized and iou(model_answer, ground_truth) > 0.5:
                correct = 1
            elif normalized and iou(model_answer, ground_truth_normalized) > 0.5:
                correct = 1
        correct_number += correct
        
        # Create a result dictionary for this example
        result = {
            'question': input_example['problem'],
            'ground_truth': ground_truth,
            'model_output': original_output,
            'extracted_answer': model_answer,
            'correct': correct
        }
        final_output.append(result)

    # Calculate and print accuracy
    accuracy = correct_number / len(data) * 100
    print(f"\nAccuracy of {ds}: {accuracy:.2f}%")

    # Save results to a JSON file
    output_path = OUTPUT_PATH.format(DATASET=ds, STEPS=steps)
    with open(output_path, "w") as f:
        json.dump({
            'accuracy': accuracy,
            'results': final_output
        }, f, indent=2)

    print(f"Results saved to {output_path}")
    print("-"*100)





