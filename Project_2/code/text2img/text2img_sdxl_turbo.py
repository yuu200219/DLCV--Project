from diffusers import AutoPipelineForText2Image
import torch
from torchmetrics.functional.multimodal import clip_score
from functools import partial
import numpy as np

def calculate_clip_score(images, prompts):
    images_int = (np.array(images, dtype=np.uint8) * 255)
    clip_score = clip_score_fn(torch.from_numpy(images_int).permute(0, 3, 1, 2), prompts).detach()
    return round(float(clip_score), 4)

pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
pipe.safety_checker = None
pipe.requires_safety_checker = False
pipe.enable_sequential_cpu_offload()

prompts = ["A view to a street from a car's front window. ISO400, high-quality, utra-hd, realistic, daytime. The pedestians passby the street. the cars stop in front of the pedestrains. The traffic light is red, so the pedestrains can pass through the street. Some modern architectures along the street.", 
          "A view to a street from a car's front window. ISO400, high-quality, utra-hd, utra-detail, realistic, daytime, hyperrealistic. The pedestrians walk along the pavement. The buildings, street lamps and trees are along the street. The sky is blue. There're some road signs beside the traffic light and street.",
          "A view to a street from a car's front window. ISO400, high-quality, utra-hd, realistic, hyperrealistic,  daytime. The buildings, street lamps and trees are along the street. The traffic light is green. The scene is at the intersection, cars, trucks, vans are cross the intersection.",
          "A view to a street from a car's front window. ISO400, high-quality, utra-hd, realistic, hyperrealistic,  daytime. A mejastic museum at the right side of the street. There's few pedestrians on the sidewalk. Some parking slot along the raod. A police man is on the sidewalk. There's a bus stop at the bus stand, people get off the bus.",
          "A view to a street from a car's front window. Real world style. ISO400, high-quality, utra-hd, realistic, hyperrealistic,  daytime. There's multiple shop, supermarket, convience store, etc. Some road sign along the sidewalk. the pedestrians fill the sidewalk. the shop sign is colorful.",
          "A view to a street from a car's front window. ISO400, high-quality, utra-hd, realistic, hyperrealistic,  daytime. A lot of trees, and the road is pave with leaves. The street lamp is along the road, no pedestrains.",
          "A view to a street from a car's front window. Real world style. ISO400, high-quality, utra-hd, realistic, hyperrealistic,  daytime. Person is going to ride on the taxi. Some cars, hotel, buildings. Trees along the road, and the railing along the sidewalk. Some pedestrians walk on the sidewalk",
          "A view to a street from a car's front window. ISO400, high-quality, utra-hd, realistic, hyperrealistic,  daytime. There're a vanding machine, and an ATM. The motocycle stop at the intersection, and the traffic light is green. A little graffiti on the building's wall, not too much. Few pedestrians.",
          "A view to a street from a car's front window. Real world style. ISO400, high-quality, utra-hd, realistic, hyperrealistic,  daytime. The bicycle is on the sidewalk. The traffic light is red. The trees are along the street. The buildings at both side of street. The cars stop before the pedestrians", 
          "A view to a street from a car's front window. ISO400, high-quality, utra-hd, realistic, hyperrealistic,  daytime. The bus stop at the bus stand, people ride on a bus. A lots of pedestsrians crowd on the sidewalk. The shop sign is on building. Some trash can along the sidewalk."]

images = []
for i in range(10):
    image = pipe(prompts[i], num_inference_step=1, guidance_scale=0.0).images[0] # don't use guidance_scale  
    images.append(image)
    image.save(f"../text2img_sdxl_turbo_output_v1/cityscape_{i}.png")
    
    torch.cuda.empty_cache()

clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")
sd_clip_score = calculate_clip_score(images, prompts)
print(f"CLIP score: {sd_clip_score}")
