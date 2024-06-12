import detectron2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.evaluation import PascalVOCDetectionEvaluator, inference_on_dataset
import os, json, cv2, random
from detectron2.data.datasets import register_pascal_voc

# REGISTER DATASETS
cls_names = ('truck', 'car', 'rider', 'person', 'train', 'motorcycle', 'bicycle', 'bus')
register_pascal_voc("my_dataset", '/mnt/c/Users/user/OneDrive - 國立中正大學/lesson/112-2/ML_CV/Exercise_1/datasets/Cityscapes_dataset/Cityscapes_dataset/VOC2007', "trainval", 2007, cls_names)
register_pascal_voc("my_test", '/mnt/c/Users/user/OneDrive - 國立中正大學/lesson/112-2/ML_CV/Exercise_1/datasets/Cityscapes_dataset/Cityscapes_dataset/VOC2007', "test", 2007, cls_names)

from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer

# SET MODEL
cfg = get_cfg()
cfg.merge_from_file("./PascalVOC-Detection/my_resnet.yaml")
cfg.DATASETS.TRAIN = ("my_dataset",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 1
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("PascalVOC-Detection/faster_rcnn_R_50_FPN.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2 # This is the real "batch size" commonly known to deep learning people
cfg.SOLVER.BASE_LR = 0.0002  # pick a good LR
cfg.SOLVER.MAX_ITER = 90000    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 8  # 8 classes, class_names = ('truck', 'car', 'rider', 'person', 'train', 'motorcycle', 'bicycle', 'bus')

# wandb
# import wandb
# wandb.login(relogin=True, key='78f66d0691441fbb503f17c6de791883d0e54f94')
# wandb.init(
#             # set the wandb project where this run will be logged
#             project="Exercise_1", 
#             name="VGG16-v2",
#             notes="20240407_v2",
#             sync_tensorboard=True
#         )

# TRAIN
cfg.OUTPUT_DIR='./output/resnet'
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
predictor = DefaultPredictor(cfg)

# EVALUATE
evaluator = PascalVOCDetectionEvaluator("my_test")
val_loader = build_detection_test_loader(cfg, "my_test")
print(inference_on_dataset(predictor.model, val_loader, evaluator))