import PIL.Image
import detectron2, sys
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.evaluation import PascalVOCDetectionEvaluator, inference_on_dataset
import os, json, cv2, random, PIL, json
from tqdm import tqdm
from os import listdir
from os.path import isfile, join
import numpy as np
from detectron2.data.datasets import register_pascal_voc, convert_to_coco_json
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer

def convert_to_coco(predictions):
    coco_data = {
        "info": {
            "description": "Convert predicted annotation to COCO format",
            "year": 2024,
            "date_created": "2024/5/12"
        },
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    category_mapping = {}  # Map class IDs to category IDs
    
    annotation_id = 1  # Counter for annotation IDs
    
    for prediction in tqdm(predictions, desc="Convert to COCO format"):
        image_info = {
            "id": prediction["image_id"],
            "width": prediction["width"],
            "height": prediction["height"],
            "file_name": prediction["file_name"]
        }
        
        coco_data["images"].append(image_info)
        for bbox, class_id, score in zip(prediction["bboxes"], prediction["class_ids"], prediction["scores"]):
            # Add category to category mapping if not already added
            if class_id not in category_mapping:
                
                category_mapping[class_id] = len(category_mapping) + 1  # Increment category ID
                coco_data["categories"].append({
                    "id": category_mapping[class_id],
                    "name": MetadataCatalog.get("my_test").thing_classes[class_id],
                    "supercategory": "object"
                })
            
            annotation = {
                "id": annotation_id,
                "image_id": prediction["image_id"],
                "category_id": category_mapping[class_id],
                "bbox": bbox,  # Convert numpy array to list
                "area": bbox[2]*bbox[3],  # Calculate area (width * height)
                "iscrowd": 0,
                "score": score
            }
            
            coco_data["annotations"].append(annotation)
            annotation_id += 1
    # print(category_mapping)
    return coco_data

cls_names = ('truck', 'car', 'rider', 'person', 'train', 'motorcycle', 'bicycle', 'bus')
# register_pascal_voc("my_dataset", "./dreambooth_output", 2024, cls_names)
register_pascal_voc("my_test", "./dreambooth_output", "test", 2007, cls_names)

cfg = get_cfg()
cfg.merge_from_file("../../../Sources/detectron2/configs/PascalVOC-Detection/my_resnet.yaml")
# cfg.DATASETS.TRAIN = ("my_dataset",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 1
cfg.SOLVER.IMS_PER_BATCH = 2 # This is the real "batch size" commonly known to deep learning people
cfg.SOLVER.BASE_LR = 0.0002  # pick a good LR
cfg.SOLVER.MAX_ITER = 90000    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 8  # 8 classes, class_names = ('truck', 'car', 'rider', 'person', 'train', 'motorcycle', 'bicycle', 'bus')

cfg.OUTPUT_DIR='../../../Sources/detectron2/configs/output/resnet-50'
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final_resnet50.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
predictor = DefaultPredictor(cfg)
# ----------
# INFERENCE
# ----------
# VISUALIZE TEH PREDICT RESULT
# im_1 = cv2.imread('./dreambooth_output/0-2b853609c92d9f004898b36dcb0acd002d827428.jpg')
# im_2 = cv2.imread('./dreambooth_output/0-9a138d285fdde953396c22913ff5103e706f0c9f.jpg')
# im_3 = cv2.imread('./dreambooth_output/0-11e5228e36d947ab9f2d1adc969460489e9c7094.jpg')
# image_path = ['./dreambooth_output/0-2b853609c92d9f004898b36dcb0acd002d827428.jpg',
#               './dreambooth_output/0-9a138d285fdde953396c22913ff5103e706f0c9f.jpg',
#               './dreambooth_output/0-11e5228e36d947ab9f2d1adc969460489e9c7094.jpg']
mypath="../../output/img2img_output"
file = [ f for f in sorted(listdir(mypath)) if isfile(join(mypath,f)) ]

# im = [im_1, im_2, im_3]
im_res = []
predictions = []
print(f"Inferenced images num: {len(file)}")

for i in tqdm(range(0, len(file)), desc="Processing inference"):
    im = np.array(PIL.Image.open( join(mypath, file[i]) ))
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                    metadata=MetadataCatalog.get('my_test'),
                    scale=0.5,
                    instance_mode=ColorMode.SEGMENTATION
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    im_res.append(out.get_image()[:, :, ::-1])
    # print(outputs)
    height, width, _ = im.shape
    prediction = {
        "image_id": i,
        "file_name": file[i],
        "width": width, 
        "height": height, 
        "bboxes": outputs["instances"].pred_boxes.tensor.tolist(),
        "class_ids": outputs["instances"].pred_classes.tolist(),
        "scores": outputs["instances"].scores.tolist(),
    }
    predictions.append(prediction)
    

coco_format_predictions = convert_to_coco(predictions)

OUTPUT_DIR = "../../output/annotations/COCO_format/"
os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(join(OUTPUT_DIR, "annotations.json"), "w") as json_file:
    json.dump(coco_format_predictions, json_file)
    
OUTPUT_DIR = "../../output/annotations/Images/"
os.makedirs(OUTPUT_DIR, exist_ok=True)
for i in tqdm(range(0, len(file)), desc="saving annotation images"):
    # cv2.imwrite('./annotations/Images/'+file[i]+'_annotation.jpg', im_res[i])
    PIL.Image.fromarray(im_res[i]).save(OUTPUT_DIR + file[i]+'_annotation.jpg')
#     cv2.imshow('im_'+str(i+1), im_res[i])
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# outputs = predictor(im)
# v = Visualizer(im[:, :, ::-1],
#                 metadata=MetadataCatalog.get('my_test'),
#                 scale=0.5,
#                 instance_mode=ColorMode.SEGMENTATION
# )
# out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
# im_res.append(out.get_image()[:, :, ::-1])
# cv2.imshow('im_'+str(0), im_res[0])
