# README
## text2img
- 可編輯程式內的`prompts`參數，產生對應之圖片
- 最終輸出圖片都會導出至`/output/text2img_output/text2img_model_name_output/`
    - e.g., stable-diffusion-v1-5會導出至`/output/text2img_output/text2img_v1_5_output/` 
- CLIP evaluation
    - 可執行`python clip_score.py ../../output/text2img_output/text2img_model_name_output`，對輸出的圖片進行CLIP Score評分。
## img2img
- 使用accelerate套件
- 使用dreambooth訓練
- 以下command執行img2img的程序
    ```bash
    export MODEL_NAME="path/to/model
    export INSTANCE_DIR="path/to/real/img/datasets/"
    export OUTPUT_DIR="path/to/saved/model"
    export DREAMBOOTH_OUTPUT="path/to/fake/img"

    accelerate launch train_dreambooth.py \
    --pretrained_model_name_or_path=$MODEL_NAME  \
    --instance_data_dir=$INSTANCE_DIR \
    --output_dir=$OUTPUT_DIR \
    --instance_prompt="A view to a street from a car's front window. ISO400, high-quality, utra-hd, realistic, hyperrealistic" \
    --resolution=512 \
    --train_batch_size=1 \
    --gradient_accumulation_steps=1 \
    --learning_rate=5e-6 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --max_train_steps=400 \
    --with_prior_preservation \
    --prior_loss_weight=1.0 \
    --class_data_dir=$DREAMBOOTH_OUTPUT \
    --class_prompt="The pedestians passby the street. the cars stop in front of the pedestrains. The traffic light is red or green. Some modern architectures along the street." \
    --snr_gamma=5.0
    ```
- 由於trained model after diffusion process過大，故不附上。如有需求可以跑上述的command，即可得出對應model。
- FID evaluation
    - 執行`python fid_score.py path/to/real/img path/to/fake/img`，比對real img與fake img，計算FID Score。

## generating_annotations
- 執行`python inference.py`，產出annotatoins至`../../output/annotatoins`
- 其中`annotations/Images`存inference的圖，`annotations/COCO_format`存annotations in COCO-format。