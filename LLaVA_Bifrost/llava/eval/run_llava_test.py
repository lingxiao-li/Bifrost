import torch.nn.functional as F
import argparse
import torch
import json
import os
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
import re


def computeIoU(bbox1, bbox2):
    x1, y1, x2, y2 = bbox1
    x3, y3, x4, y4 = bbox2
    intersection_x1 = max(x1, x3)
    intersection_y1 = max(y1, y3)
    intersection_x2 = min(x2, x4)
    intersection_y2 = min(y2, y4)
    intersection_area = max(0, intersection_x2 - intersection_x1 + 1) * \
        max(0, intersection_y2 - intersection_y1 + 1)
    bbox1_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    bbox2_area = (x4 - x3 + 1) * (y4 - y3 + 1)
    union_area = bbox1_area + bbox2_area - intersection_area
    iou = intersection_area / union_area
    return iou



def image_parser(args):
    out = args.image_file.split(args.sep)
    return out


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out


def eval_model(tokenizer, model, image_processor, model_name, args):
    # Model

    qs = args.query
    image_token_se = DEFAULT_IM_START_TOKEN + \
        DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, args.conv_mode, args.conv_mode
            )
        )
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    image_files = image_parser(args)
    images = load_images(image_files)
    image_sizes = [x.size for x in images]
    images_tensor = process_images(
        images,
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)

    input_ids = (
        tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
        )

    outputs = tokenizer.batch_decode(
        output_ids, skip_special_tokens=True)[0].strip()
    # print(outputs)
    return outputs


def create_model(args):
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name
    )
    return tokenizer, model, image_processor, model_name


if __name__ == "__main__":
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()
    '''
    image_dir  = "/home/zhangyuanyuan/Lingxiao/datasets/llava_fine_tune_data/images"
    test_data_path = "/home/zhangyuanyuan/Lingxiao/datasets/llava_fine_tune_data/validation/dataset.json"
    with open(test_data_path, 'r') as f:
        data_dict = json.load(f)


    def str_int(output):
        output = output.strip('[').strip(']')
        output = list(output.split(','))
        output = [float(i) for i in output]
        return output


    # print(data_dict[0])
    box_loss = 0
    depth_loss = 0
    mode = 'ours'
    model_path = "PATH_TO/LLaVA/llava/checkpoints/llava-v1.5-7b-task-lora-full-data-v4"
    model_base = "PATH_TO/.cache/huggingface/hub/liuhaotian/llava-v1.5-7b"
    args = type('Args', (), {
        "model_path": model_path,
        "model_base": model_base,
        "model_name": get_model_name_from_path(model_path),
        "sep": ",",
        "temperature": 0.2,
        "top_p": None,
        "num_beams": 1,
        "max_new_tokens": 512
    })()
    tokenizer, model, image_processor, model_name = create_model(args)
    for item in tqdm(data_dict):
        image_path = item['image']
        conversations = item['conversations']
        prompt = conversations[0]['value']
        target = conversations[1]['value']
        target_box = target.split(']')[0]
        target_depth = target.split(']')[1]
        image_file = os.path.join(image_dir, image_path)


        args = type('Args', (), {
            "model_path": model_path,
            "model_base": model_base,
            "model_name": get_model_name_from_path(model_path),
            "query": prompt,
            "conv_mode": None,
            "image_file": image_file,
            "sep": ",",
            "temperature": 0.2,
            "top_p": None,
            "num_beams": 1,
            "max_new_tokens": 512
        })()
        outputs = eval_model(tokenizer, model, image_processor, model_name, args)
        # print(outputs)
        if mode == 'ours':
            predicted_box = outputs.split(']')[0].split('[')[1]
            predicted_depth = outputs.split(']')[1].split(', ')[1][:-1]
            # print("predicted_box: " + predicted_box)
            # print("predicted_depth: " + predicted_depth)
            target_box = str_int(target_box)
            # print("target_box: " + str(target_box))
            target_depth = str(target_depth).split(', ')[1][:-1]
            # print("target_depth: " + target_depth)
            # target_box = torch.Tensor(str_int(str(target_box)))
            # predicted_box = torch.Tensor(str_int(predicted_box))
            # box_loss += F.mse_loss(predicted_box, target_box)
            predicted_box = str_int(predicted_box)
            bbox_pred = str(predicted_box).strip('[]').split(',')
            bbox_pred = [float(x) for x in bbox_pred]
            target_box = str_int(str(target_box))
            bbox_target = str(target_box).strip('[]').split(',')
            bbox_target = [float(x) for x in bbox_target]
            iou = computeIoU(bbox_pred, bbox_target)
            print("iou: "+str(iou))
            if iou > 0.5:
                box_loss += 1
            predicted_depth = torch.Tensor(str_int(predicted_depth))
            target_depth = torch.Tensor(str_int(target_depth))
            depth_loss += F.mse_loss(predicted_depth, target_depth)
        else:
            predicted_box = outputs.split(']')[0].split('[')[1]
            # predicted_depth = outputs.split(']')[1].split(', ')[1][:-1]
            # print("predicted_box: " + predicted_box)
            # print("predicted_depth: " + predicted_depth)
            target_box = str_int(target_box)
            # print("target_box: " + str(target_box))
            # target_depth = str(target_depth).split(', ')[1][:-1]
            # print("target_depth: " + target_depth)
            # target_box = torch.Tensor(str_int(str(target_box)))
            # predicted_box = torch.Tensor(str_int(predicted_box))
            # box_loss += F.mse_loss(predicted_box, target_box)
            # predicted_depth = torch.Tensor(str_int(predicted_depth))
            # target_depth = torch.Tensor(str_int(target_depth))
            # depth_loss += F.mse_loss(predicted_depth, target_depth)
            predicted_box = str_int(predicted_box)
            bbox_pred = str(predicted_box).strip('[]').split(',')
            bbox_pred = [float(x) for x in bbox_pred]
            target_box = str_int(str(target_box))
            bbox_target = str(target_box).strip('[]').split(',')
            bbox_target = [float(x) for x in bbox_target]
            iou = computeIoU(bbox_pred, bbox_target)
            print("iou: "+str(iou))
            if iou > 0.5:
                box_loss += 1
            
        # print("box_loss: "+str(box_loss))
        # print("depth_loss: "+str(depth_loss))
    mean_box_loss = box_loss / len(data_dict)
    mean_depth_loss = depth_loss / len(data_dict)
    print("mean_box_loss: "+str(mean_box_loss))
    print("mean_depth_loss: "+str(mean_depth_loss))
