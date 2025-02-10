import os
import glob
from tqdm import tqdm
from run_monodepth_api import run, initialize_dpt_model
import os
import numpy as np
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"

dpt_model, transform = initialize_dpt_model(model_path='/home/mhf/dxl/Lingxiao/Codes/DPT/weights/dpt_large-midas-2f21e586.pt')

input_path = "/data2/mhf/DXL/Lingxiao/datasets/COCO_dataset_2017/train2017"
output_path = "/data2/mhf/DXL/Lingxiao/datasets/COCO_dataset_2017/train2017_depth"
# print(input_path)
# print(output_path)
run(dpt_model, transform, input_path, output_path)