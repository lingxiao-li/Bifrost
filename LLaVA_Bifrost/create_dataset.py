import json
import torchvision
from PIL import ImageDraw
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import torchvision
from PIL import ImageDraw
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from Inpaint_Anything.remove_anything_api import remove_any_thing, create_sam_model
from DPT.run_monodepth_api import run, initialize_dpt_model
from PIL import Image
from io import BytesIO
import requests
import os
import uuid
import cv2
import numpy as np
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

cat_2014 = 'PATH_TO/datasets/COCO_dataset_2017/annotations/instances_train2017.json'
cat_2017 = 'PATH_TO/datasets/COCO_dataset_2017/annotations/instances_val2017.json'
relation = {'left of', 'right of', 'in front of', 'behind', 'on top of',
            'under', 'next to', 'surround', 'near', 'above', 'below'}
output_path = 'PATH_TO/datasets/llava_fine_tune_data'


def collect_category(json_path):
    cat_dic = [[0] * 2 for _ in range(90)]
    json_file = json_path
    if json_file is not None:
        with open(json_file, 'r') as COCO:
            js = json.loads(COCO.read())
    for i in js['categories']:
        cat_dic[i['id']-1][0] = i['supercategory']
        cat_dic[i['id']-1][1] = i['name']
    return cat_dic


cat_2017 = collect_category(cat_2017)

# import coco 2017 images and annotations
# replace validation dataset with COCO training dataset to create customized training dataset
coco_dataset = torchvision.datasets.CocoDetection(root="PATH_TO/datasets/COCO_dataset_2017/val2017",
                                                  annFile="PATH_TO/datasets/COCO_dataset_2017/annotations/instances_val2017.json")
if __name__ == "__main__":
    # initialize the models
    dpt_model, transform = initialize_dpt_model(
        model_path='PATH_TO/Codes/DPT/weights/dpt_large-midas-2f21e586.pt')
    SAM_predictor = create_sam_model()
    # Initialize list to hold all JSON data
    json_data_list = []

    # ImageDraw handler
    # image_handler = ImageDraw.ImageDraw(image)
    for image, info in tqdm(coco_dataset):
        original_image = image.copy()
        # image_handler = ImageDraw.ImageDraw(image)
        if len(info) > 0:
            obj_num = min(random.randint(2, 5), len(info))
            selected_objs = []
            selected_cat = []
            random_num = -1
            for annotation in info:
                x_min_select, y_min_select, width_select, height_select = annotation['bbox']
                x_min_scale_select = x_min_select/image.size[0]
                y_min_scale_select = y_min_select/image.size[1]
                width_scale_select = width_select/image.size[0]
                height_scale_select = height_select/image.size[1]
                if cat_2017[annotation['category_id']-1][1] in selected_cat:
                    continue
                elif width_scale_select*height_scale_select < 0.2:
                    continue
                else:
                    selected_cat.append(
                        cat_2017[annotation['category_id']-1][1])
                    selected_objs.append(annotation)
                    # if len(selected_objs) == obj_num+5:
                    #     break
                # bbox is the position of the detection box
                # print(annotation['category_id'], cat_201
            # print(len(selected_objs))
            obj_relations = {}
            if len(selected_objs) > 0:
                selected_object = random.sample(selected_objs, 1)
                x_min_select, y_min_select, width_select, height_select = selected_object[0]['bbox']
                x_min_scale_select = x_min_select/image.size[0]
                y_min_scale_select = y_min_select/image.size[1]
                width_scale_select = width_select/image.size[0]
                height_scale_select = height_select/image.size[1]
                selected_object_name = cat_2017[selected_object[0]
                                                ['category_id']-1][1]
                random.shuffle(selected_objs)
                # image_handler.rectangle(
                #     ((x_min_select, y_min_select), (x_min_select + width_select, y_min_select + height_select)), fill='blue')
                # get the mask of the selected object
                SAM_predictor.set_image(np.array(original_image))
                point_coords = np.array([float(x_min_select + width_select/2), float(y_min_select + height_select/2)])
                point_labels = np.array([1])
                masks, _, _ = SAM_predictor.predict(point_coords=point_coords,
                                                    point_labels=point_labels,
                                                    multimask_output=True)
                # save the mask image
                back_mask = masks[1].astype(np.uint8)
                plt.imshow(back_mask)
                if np.random.uniform(0, 1) < 0.1:
                    # Instruction for raplace objects
                    unique_id = str(uuid.uuid4())
                    # Define image path
                    image_path = os.path.join(
                        output_path+'/images', f"{unique_id}.jpg")
                    image_path_2 = os.path.join(
                        output_path+'/temp', f"{unique_id}.png")
                    image_path_3 = os.path.join(
                        output_path+'/depth', f"{unique_id}.png")
                    cv2.imwrite(image_path, cv2.cvtColor(
                        np.array(original_image), cv2.COLOR_RGB2BGR))
                    cv2.imwrite(image_path_2, cv2.cvtColor(
                        np.array(original_image), cv2.COLOR_RGB2BGR))
                    

                    # predict the depth of the selected object in the image

                    run(dpt_model, transform, output_path +
                        '/temp', output_path+'/depth')
                    os.remove(image_path_2)
                    image_depth = cv2.imread(image_path_3, cv2.IMREAD_UNCHANGED)
                    result = np.zeros(image_depth.shape, dtype=np.float32)
                    cv2.normalize(image_depth, result, alpha=0, beta=1,
                                norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                    result_bbox = np.zeros((int(height_select), int(width_select)), dtype=np.float32)
                    result_bbox = result[int(y_min_select):int(y_min_select+height_select), int(x_min_select):int(x_min_select+width_select)]
                    max_depth = np.max(result_bbox)
                    med_depth = np.median(result_bbox)
                    min_depth = np.min(result_bbox)
                    instruction = "Replace the " + \
                        str(selected_object_name) + " with the reference image"
                    instruction += ', output the bounding box and the max, median and min depth values of the object.'
                    answer = "[%.3f, %.3f, %.3f, %.3f], " % (
                        x_min_scale_select, y_min_scale_select, width_scale_select+x_min_scale_select, height_scale_select+y_min_scale_select)
                    answer += "%.3f, " % max_depth
                    answer += "%.3f, " % med_depth
                    answer += "%.3f." % min_depth
                    obj_relations["replace"] = selected_object_name
                    # print("Instructions: ")
                    # print(instruction)
                    # print("Answer: ")
                    # print(answer)

                    if len(obj_relations) > 0:
                        # Remove duplicates and format answers
                        formatted_answers = answer

                        # Structure for LLaVA JSON
                        json_data = {
                            "id": unique_id,
                            "image": f"{unique_id}.jpg",
                            "conversations": [
                                {
                                    "from": "human",
                                    "value": instruction
                                },
                                {
                                    "from": "gpt",
                                    "value": formatted_answers
                                }
                            ]
                        }

                        # Append to list
                        json_data_list.append(json_data)

                # Define the output folder

                # Create a unique ID for each image
                unique_id = str(uuid.uuid4())
                # Define image path
                image_path = os.path.join(
                    output_path+'/images', f"{unique_id}.jpg")
                image_path_2 = os.path.join(
                    output_path+'/temp', f"{unique_id}.png")
                image_path_3 = os.path.join(
                    output_path+'/depth', f"{unique_id}.png")
                # Instruction for place new objects
                obj_relations = {}
                cv2.imwrite(image_path_2, cv2.cvtColor(
                    np.array(original_image), cv2.COLOR_RGB2BGR))
                # predict the depth of the selected object in the image
                run(dpt_model, transform, output_path + '/temp', output_path+'/depth')
                os.remove(image_path_2)
                image_depth = cv2.imread(image_path_3, cv2.IMREAD_UNCHANGED)
                result = np.zeros(image_depth.shape, dtype=np.float32)
                cv2.normalize(image_depth, result, alpha=0, beta=1,
                            norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                result_bbox = np.zeros((int(height_select), int(width_select)), dtype=np.float32)
                result_bbox = result[int(y_min_select):int(y_min_select+height_select), int(x_min_select):int(x_min_select+width_select)]
                max_depth = np.max(result_bbox)
                med_depth = np.median(result_bbox)
                min_depth = np.min(result_bbox)
                for annotation in selected_objs:
                    # bbox is the position of the detection box
                    if cat_2017[annotation['category_id']-1][1] == cat_2017[selected_object[0]['category_id']-1][1]:
                        continue
                    else:
                        x_min, y_min, width, height = annotation['bbox']
                        x_min_scale = x_min/image.size[0]
                        y_min_scale = y_min/image.size[1]
                        width_scale = width/image.size[0]
                        height_scale = height/image.size[1]
                        # image_handler.rectangle(
                        #     ((x_min, y_min), (x_min + width,  y_min + height)), fill='red')

                        # if the two objects are too far away, ignore the relation
                        if x_min_scale_select - x_min_scale - width_scale > 0.3 or x_min_scale - x_min_scale_select - width_scale_select > 0.3 or y_min_scale_select - y_min_scale - height_scale > 0.3 or y_min_scale - y_min_scale_select - height_scale_select > 0.3:
                            continue
                        # print("center: " + str(x_min + width/2) + " " + str(y_min + height/2))
                        # print(x_min_scale, y_min_scale, width_scale, height_scale)
                        else:
                            selected_relation = None
                            if x_min_scale_select - x_min_scale - width_scale < 0 and x_min_scale_select + width_scale_select - x_min_scale > 0:
                                # print("1")
                                if y_min_scale + height_scale < y_min_scale_select + 0.01:
                                    '''
                                    if result[int((y_min_select + height_select/2))][int((x_min_select + width_select/2))] - \
                                            result[int((y_min + height/2))][int((x_min + width/2))] < 0.2 and \
                                            result[int((y_min_select + height_select/2))][int((x_min_select + width_select/2))] - \
                                            result[int((y_min + height/2))][int((x_min + width/2))] > -0.2:
                                            '''
                                    if y_min_scale_select - y_min_scale - height_scale < 0.01:
                                        # print("2")
                                        selected_relation = random.choice(
                                            ['below', 'under', 'beneath', 'underneath', 'under', 'under'])
                                    else:
                                        # print("3")
                                        selected_relation = random.choice(
                                            ['below', 'beneath', 'underneath'])
                                elif y_min_scale + 0.01 > y_min_scale_select + height_scale_select:
                                    '''
                                    if result[int((y_min_select + height_select/2))][int((x_min_select + width_select/2))] - \
                                            result[int((y_min + height/2))][int((x_min + width/2))] < 0.2 and \
                                            result[int((y_min_select + height_select/2))][int((x_min_select + width_select/2))] - \
                                            result[int((y_min + height/2))][int((x_min + width/2))] > -0.2:
                                            '''
                                    if y_min_select + height_select - y_min > -0.01:
                                        # print("4")
                                        selected_relation = random.choice(
                                            ['above', 'on top of', 'at the top of', 'over', 'on', 'on'])
                                    else:
                                        # print("5")
                                        selected_relation = random.choice(
                                            ['above', 'over'])
                                elif x_min_scale_select - x_min_scale - width_scale < -0.06 and x_min_scale_select + width_scale_select - x_min_scale > 0.06:
                                    # if result[int((y_min_select + height_select/2))][int((x_min_select + width_select/2))] - \
                                    #         result[int((y_min + height/2))][int((x_min + width/2))] > 0.1:
                                    if y_min_scale_select > y_min_scale and y_min_scale + height_scale > y_min_scale_select:
                                        # print("6")
                                        selected_relation = random.choice(
                                            ['in front of'])
                                    elif y_min_scale_select + height_scale_select < y_min_scale + height_scale and y_min_scale_select + height_scale_select > y_min_scale:
                                        # if result[int((y_min_select + height_select/2))][int((x_min_select + width_select/2))] - \
                                        #         result[int((y_min + height/2))][int((x_min + width/2))] < -0.1:
                                            # print("7")
                                        selected_relation = random.choice(
                                            ['behind'])
                            elif y_min_scale_select - y_min_scale - height_scale < 0.06 and y_min_scale_select + height_scale_select - y_min_scale > -0.06:
                                '''
                                if result[int((y_min_select + height_select/2))][int((x_min_select + width_select/2))] - \
                                        result[int((y_min + height/2))][int((x_min + width/2))] < 0.1 and \
                                        result[int((y_min_select + height_select/2))][int((x_min_select + width_select/2))] - \
                                        result[int((y_min + height/2))][int((x_min + width/2))] > -0.1:
                                        '''
                                if x_min > x_min_select + width_select and x_min_scale - x_min_scale_select - width_scale_select < 0.2:
                                    if x_min_scale - x_min_scale_select - width_scale_select < 0.01:
                                        # print("8")
                                        selected_relation = random.choice(
                                            ['next to', 'near', 'beside', 'alongside', 'to the left of', 'to the left of'])
                                    else:
                                        # print("9")
                                        selected_relation = 'to the left of'
                                elif x_min + width < x_min_select and x_min_scale_select - x_min_scale - width_scale < 0.2:
                                    if x_min_scale_select - x_min_scale - width_scale < 0.01:
                                        # print("10")
                                        selected_relation = random.choice(
                                            ['next to', 'near', 'beside', 'alongside', 'to the right of', 'to the right of'])
                                    else:
                                        # print("11")
                                        selected_relation = 'to the right of'
                                elif x_min_scale_select - x_min_scale - width_scale > 0.3 or x_min_scale - x_min_scale_select - width_scale_select > 0.3 or y_min_scale_select - y_min_scale - height_scale > 0.3 or y_min_scale - y_min_scale_select - height_scale_select > 0.3:
                                    selected_relation = None
                            # check redundant relations
                            if selected_relation is not None:
                                if selected_relation in obj_relations.keys():
                                    if selected_relation in ['below', 'under', 'beneath', 'underneath', 'in front of']:
                                        if y_min_scale + height_scale > obj_relations[selected_relation][1][1] + obj_relations[selected_relation][1][3]:
                                            obj_relations[selected_relation] = [
                                                cat_2017[annotation['category_id']-1][1], (x_min_scale, y_min_scale, width_scale, height_scale)]
                                    elif selected_relation in ['above', 'on top of', 'at the top of', 'over', 'on', 'behind']:
                                        if y_min_scale < obj_relations[selected_relation][1][1]:
                                            obj_relations[selected_relation] = [
                                                cat_2017[annotation['category_id']-1][1], (x_min_scale, y_min_scale, width_scale, height_scale)]
                                    elif selected_relation in ['to the left of']:
                                        if x_min_scale < obj_relations[selected_relation][1][0]:
                                            obj_relations[selected_relation] = [
                                                cat_2017[annotation['category_id']-1][1], (x_min_scale, y_min_scale, width_scale, height_scale)]
                                    elif selected_relation in ['to the right of']:
                                        if x_min_scale + width_scale > obj_relations[selected_relation][1][0] + obj_relations[selected_relation][1][2]:
                                            obj_relations[selected_relation] = [
                                                cat_2017[annotation['category_id']-1][1], (x_min_scale, y_min_scale, width_scale, height_scale)]
                                else:
                                    obj_relations[selected_relation] = [
                                        cat_2017[annotation['category_id']-1][1], [x_min_scale, y_min_scale, width_scale, height_scale]]

                            if len(obj_relations) == obj_num:
                                break

                # print(obj_relations)
                if len(obj_relations) > 0:
                    # remove the selected object in the image
                    remove_any_thing(SAM_predictor, original_image, [float(x_min_select + width_select/2), float(
                        y_min_select + height_select/2)], output_path + '/images')

                    inpaint_image = Image.open(
                        output_path + '/images/inpainted_with_mask_0.png')
                    if inpaint_image.mode == "RGBA":
                        inpaint_image = inpaint_image.convert("RGB")
                    # Save image
                    inpaint_image.save(image_path)

                    # format the instruction and answer
                    instruction = "Place the " + \
                        str(selected_object_name) + " "
                    count = 0
                    for relation in obj_relations.keys():
                        instruction += relation + " the " + \
                            obj_relations[relation][0]
                        count += 1
                        if count != len(obj_relations):
                            instruction += " and "
                    instruction += ', output the bounding box and the max, median and min depth values of the object.'
                    # print("Instructions: ")
                    # print(instruction)
                    # print("Answer: ")
                    answer = "[%.3f, %.3f, %.3f, %.3f], " % (
                        x_min_scale_select, y_min_scale_select, x_min_scale_select+width_scale_select, y_min_scale_select+height_scale_select)
                    answer += "%.3f, " % max_depth
                    answer += "%.3f, " % med_depth
                    answer += "%.3f." % min_depth
                    # print(answer)
                if len(obj_relations) > 0:
                    # Remove duplicates and format answers
                    formatted_answers = answer

                    # Structure for LLaVA JSON
                    json_data = {
                        "id": unique_id,
                        "image": f"{unique_id}.jpg",
                        "conversations": [
                            {
                                "from": "human",
                                "value": instruction
                            },
                            {
                                "from": "gpt",
                                "value": formatted_answers
                            }
                        ]
                    }

                    # Append to list
                    json_data_list.append(json_data)

        # if len(json_data_list) == 5:
        #         break

    # Save the JSON data list to a file
    json_output_path = os.path.join(output_path, "validation", 'dataset.json')
    with open(json_output_path, 'w') as json_file:
        json.dump(json_data_list, json_file, indent=4)
