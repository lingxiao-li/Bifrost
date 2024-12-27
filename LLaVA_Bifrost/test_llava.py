from PIL import Image
import requests
from transformers import AutoProcessor, LlavaForConditionalGeneration
import json
import torch.nn.functional as F 
import torch
with open('dataset.json', 'r') as f:
    data_dict = json.load(f)

def str_int(output):
    output = output.strip('[').strip(']')
    output = list(output.split(','))
    output = [float(i) for i in output]
    return output 

print(data_dict[0])
loss = 0




for item in data_dict:
    image_path = item['image']
    conversations = item['conversations']
    prompt = conversations[0]['value']
    target = conversations[1]['value']
    target_box = target.split(']')[0]
    target_depth = target.split(']')[1]

    
    prompt = f"USER: <image>\n{prompt} ASSISTANT:"
        
    model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
    processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

    # prompt = "USER: <image>\nWhat's the content of the image? ASSISTANT:"
    # url = "https://www.ilankelman.org/stopsigns/australia.jpg"
    # image = Image.open(requests.get(url, stream=True).raw)
    image = Image.open(image_path)


    inputs = processor(text=prompt, images=image, return_tensors="pt")

    # Generate
    generate_ids = model.generate(**inputs, max_new_tokens=50)
    output = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    # evaluate 
    output = output.split('ASSISTANT: ')[1]
    output = str_int(output)

    target_box = str_int(target_box)
    output = torch.Tensor(output)
    target_box = torch.Tensor(target_box)
    loss += F.mse_loss(output, target_box)

mean_loss = loss / len(data_dict)

print(mean_loss)