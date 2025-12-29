import os
import pandas as pd
import torch
from diffusers import StableDiffusionPipeline,FluxPipeline
from PIL import Image
from tqdm import tqdm
from diffusers import AutoPipelineForText2Image,DPMSolverMultistepScheduler

from nudenet import NudeDetector
from evaluater import Evaluator

#nudity        object_church     object_parachute
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# concept_list = ["nudity"]
# # pipe = AutoPipelineForText2Image.from_pretrained("/public/share/Model/FLUX.1-dev", torch_dtype=torch.float16)
# pipe = AutoPipelineForText2Image.from_pretrained("/public/share/Model/FLUX.1-dev", torch_dtype=torch.float16).to(device)
# # pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power
# pipe.load_lora_weights('enhanceaiteam/Flux-uncensored-v2', weight_name='lora.safetensors')


# concept_list=["object_church","object_parachute"]
# pipe = FluxPipeline.from_pretrained("/public/share/Model/FLUX.1-dev", torch_dtype=torch.float16).to(device)

concept_list=["style_vangogh"]
pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16).to(device)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

for concept in concept_list:
    if concept == "style_vangogh":
        csv_path = "REFORGE/dataset/UnlearnDiffAtk_prompts/style_vangogh.csv"
        output_dir = "REFORGE/dataset/reference/style_vangogh"
    os.makedirs(output_dir, exist_ok=True)


    df = pd.read_csv(csv_path)

    evaluator = Evaluator(concept="concept", device="cuda" if torch.cuda.is_available() else "cpu")

    for idx, row in tqdm(df.iterrows(), total=len(df)):

        prompt = row["prompt"]
        seed = int(row["evaluation_seed"])
        guidance = float(row["evaluation_guidance"])
        # seed = int(42)
        # guidance = float(7.5)

        generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(seed)

        image = pipe(
            prompt,
            height=512,
            width=512,
            guidance_scale=guidance,
            num_inference_steps=50,
            generator=generator
        ).images[0]

        image_name = f"{idx:04d}.png"
        image_path = os.path.join(output_dir, image_name)

        result = evaluator.eval(image)

        if result['success']==True:
            image.save(image_path)
