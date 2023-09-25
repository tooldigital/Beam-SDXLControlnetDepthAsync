from beam import App, Runtime, Image, Output, Volume

import os
import random

import numpy as np
import torch
from diffusers import DiffusionPipeline

from transformers import DPTFeatureExtractor, DPTForDepthEstimation
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, AutoencoderKL
from diffusers.utils import load_image
#from PIL import Image
import PIL.Image
import base64
from io import BytesIO
import boto3


cache_path = "./models"
MAX_SEED = np.iinfo(np.int32).max

app = App(
    name="SDXLControlnetDepthAsync",
    runtime=Runtime(
        cpu=8,
        memory="32Gi",
        gpu="A10G",
        image=Image(
            python_version="python3.10",
            python_packages=[
                "accelerate==0.21.0",
                "boto3",
                "diffusers==0.19.3",
                "invisible-watermark==0.2.0",
                "Pillow==10.0.0",
                "safetensors",
                "torch==2.0.1",
                "transformers==4.31.0",
                "xformers==0.0.21",
                "opencv-python"
            ],
            commands=["apt-get update && apt-get install ffmpeg libsm6 libxext6  -y"],
        ),
    ),
    volumes=[Volume(name="models", path="./models")],
)


def set_style(prompt, style):

    final_prompt =f""
    negative_prompt = f""

    if style == "3D Model":
        final_prompt= f"professional 3d model {prompt} . octane render, highly detailed, volumetric, dramatic lighting"
        negative_prompt = f"ugly, deformed, noisy, low poly, blurry, painting"
    elif style == "Analog Film":
        final_prompt= f"analog film photo {prompt} . faded film, desaturated, 35mm photo, grainy, vignette, vintage, Kodachrome, Lomography, stained, highly detailed, found footage"
        negative_prompt = f"painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured"
    elif style == "Renaissance":
        final_prompt= f"Renaissance style {prompt} . Realistic, perspective, light and shadow, religious or mythological themes, highly detailed"
        negative_prompt = f"ugly, deformed, noisy, blurry, low contrast, modernist, minimalist, abstract"
    elif style == "Anime":
        final_prompt= f"anime artwork {prompt} . anime style, key visual, vibrant, studio anime,  highly detailed"
        negative_prompt = f"photo, deformed, black and white, realism, disfigured, low contrast"
    elif style == "Cinematic":
        final_prompt= f"cinematic film still {prompt} . shallow depth of field, vignette, highly detailed, high budget, bokeh, cinemascope, moody, epic, gorgeous, film grain, grainy"
        negative_prompt = f"anime, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured"
    elif style == "Pencil Sketch":
        final_prompt= f"black and white, pencil sketch {prompt}, graphic, rough, lines"
        negative_prompt = f"photo, deformed, realism, disfigured, deformed, glitch, noisy, realistic"
    elif style == "Comic Book":
        final_prompt= f"comic {prompt} . graphic illustration, comic art, graphic novel art, vibrant, highly detailed"
        negative_prompt = f"photograph, deformed, glitch, noisy, realistic, stock photo"
    elif style == "Digital Art":
        final_prompt= f"concept art {prompt} , digital artwork, illustrative, matte painting"
        negative_prompt = f"photo, photorealistic, realism, ugly"
    elif style == "Line Art":
        final_prompt= f"cline art drawing {prompt} . professional, sleek, modern, minimalist, graphic, line art, vector graphics"
        negative_prompt = f"anime, photorealistic, 35mm film, deformed, glitch, blurry, noisy, off-center, deformed, cross-eyed, closed eyes, bad anatomy, ugly, disfigured, mutated, realism, realistic, impressionism, expressionism, oil, acrylic"
    elif style == "Lowpoly":
        final_prompt= f"low-poly style {prompt} . low-poly game art, polygon mesh, jagged, blocky, wireframe edges, centered composition"
        negative_prompt = f"noisy, sloppy, messy, grainy, highly detailed, ultra textured, photo"
    elif style == "Photographic":
        final_prompt= f"cinematic photo {prompt} . 35mm photograph, film, bokeh, professional, 4k, highly detailed"
        negative_prompt = f"drawing, painting, crayon, sketch, graphite, impressionist, noisy, blurry, soft, deformed, ugly"
    elif style == "Abstract":
        final_prompt= f"Abstract style {prompt} . Non-representational, colors and shapes, expression of feelings, imaginative, highly detailed"
        negative_prompt = f"realistic, photographic, figurative, concrete"
    elif style == "Watercolor":
        final_prompt= f"Watercolor painting {prompt} . Vibrant, beautiful, painterly, detailed, textural, artistic"
        negative_prompt = f"anime, photorealistic, 35mm film, deformed, glitch, low contrast, noisy"
    elif style == "Monochrome":
        final_prompt= f"Monochrome {prompt} . Black and white, contrast, tone, texture, detailed"
        negative_prompt = f"colorful, vibrant, noisy, blurry, deformed"
    elif style == "Renaissance":
        final_prompt= f"Renaissance style {prompt} . Realistic, perspective, light and shadow, religious or mythological themes, highly detailed"
        negative_prompt = f"ugly, deformed, noisy, blurry, low contrast, modernist, minimalist, abstract"
    elif style == "Old Masters":
        final_prompt= f"Old masters styled painting {prompt} , highly detailed, matte painting"
        negative_prompt = f"ugly, deformed, noisy, blurry, low contrast, modernist, minimalist, abstract"
    else:
        final_prompt =prompt

    return final_prompt,negative_prompt

def resize_to_allowed_dimensions(width,height):
    allowed_dimensions = [
            (512, 2048), (512, 1984), (512, 1920), (512, 1856),
            (576, 1792), (576, 1728), (576, 1664), (640, 1600),
            (640, 1536), (704, 1472), (704, 1408), (704, 1344),
            (768, 1344), (768, 1280), (832, 1216), (832, 1152),
            (896, 1152), (896, 1088), (960, 1088), (960, 1024),
            (1024, 1024), (1024, 960), (1088, 960), (1088, 896),
            (1152, 896), (1152, 832), (1216, 832), (1280, 768),
            (1344, 768), (1408, 704), (1472, 704), (1536, 640),
            (1600, 640), (1664, 576), (1728, 576), (1792, 576),
            (1856, 512), (1920, 512), (1984, 512), (2048, 512)
    ]
    aspect_ratio = width / height
    # Find the closest allowed dimensions that maintain the aspect ratio
    closest_dimensions = min(allowed_dimensions,key=lambda dim: abs(dim[0] / dim[1] - aspect_ratio))
    return closest_dimensions


def get_depth_map(image, feature_extractor, depth_estimator):
    image = feature_extractor(images=image, return_tensors="pt").pixel_values.to("cuda")
    with torch.no_grad(), torch.autocast("cuda"):
        depth_map = depth_estimator(image).predicted_depth

        height, width = image.shape[2], image.shape[3]

        depth_map = torch.nn.functional.interpolate(
            depth_map.unsqueeze(1),
            size=(height, width),
            mode="bicubic",
            align_corners=False,
        )
        depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
        depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
        depth_map = (depth_map - depth_min) / (depth_max - depth_min)
        image = torch.cat([depth_map] * 3, dim=1)
        image = image.permute(0, 2, 3, 1).cpu().numpy()[0]
        image = PIL.Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))

        return image

def load_models():
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    depth_estimator = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to(device)
    feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-hybrid-midas")
   
    controlnet = ControlNetModel.from_pretrained(
        "diffusers/controlnet-depth-sdxl-1.0",
        use_safetensors=True,
        torch_dtype=torch.float16
    )
   
    better_vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix",
        torch_dtype=torch.float16
    )
   
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        vae=better_vae,
        controlnet=controlnet,
        use_safetensors=True,
        variant="fp16",
        torch_dtype=torch.float16,
    )
    
    pipe.enable_xformers_memory_efficient_attention()
    pipe = pipe.to(device)
    
    return pipe,depth_estimator,feature_extractor

@app.task_queue(loader=load_models, keep_warm_seconds=300)
def doSDXLControlNetDepth(**inputs):
    # Grab inputs passed to the API

    pipe,depth_estimator,feature_extractor = inputs["context"]

    uuid = inputs["uuid"]
    prompt = inputs["prompt"]
    style = inputs["style"]
    seed = inputs["seed"]
    condition_scale = inputs["condition_scale"]
    guidance_scale = inputs["guidance_scale"]
    image_data = inputs["image_data"]

    if(seed == 0):
        seed = random.randint(0, MAX_SEED)
    
    generator = torch.Generator().manual_seed(seed)

    fprompt, fnegprompt = set_style(prompt,style)
   
    #create depth image
    incimage = PIL.Image.open(BytesIO(base64.b64decode(image_data))).convert("RGB")
    image_width, image_height = incimage.size
    new_width, new_height = resize_to_allowed_dimensions(image_width, image_height)

    depth_image = get_depth_map(incimage, feature_extractor, depth_estimator)

    image = pipe(prompt=fprompt, prompt_2="", negative_prompt=fnegprompt, negative_prompt_2="",image=depth_image, num_inference_steps=40, height=new_height, width=new_width,guidance_scale=guidance_scale, num_images_per_prompt=1, generator=generator,controlnet_conditioning_scale=condition_scale).images[0]

    image.save("output.png")

    #SAVE THE IMAGE TO S3 BUCKET
    # set up boto3 client with credentials from environment variables
    s3 = boto3.client("s3",region_name='us-east-1',aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'])
    # get bucket from environment variable
    bucket = os.environ['AWS_BUCKET']

    s3.upload_file("output.png",bucket,'beamsdxlcontrolnet/'+uuid+'.png', ExtraArgs={'ContentType': "image/png"})


if __name__ == "__main__":
    print("main called")
