import os
import time
import torch
import torchvision
from PIL import Image
from vqgan import VQModel
from diffuzz import Diffuzz
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, T5EncoderModel, CLIPTextModel
from torch.utils.data import DataLoader
from modules import Paella, EfficientNetEncoder, Prior, DiffNeXt, sample
from diffuzz import Diffuzz
import transformers
import time
transformers.utils.logging.set_verbosity_error()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
checkpoint_stage_a = "models/vqgan_f4_v1_500k.pt"
checkpoint_stage_b = "models/model_stage_b.pt"
checkpoint_stage_c = "models/model_stage_c_ema.pt"

def decode(vqmodel, img_seq):
    return vqmodel.decode_indices(img_seq)

def embed_clip(caption, negative_caption="", batch_size=4, device="cuda"):
    clip_model = CLIPTextModel.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K").to(device).eval().requires_grad_(False)
    clip_tokenizer = AutoTokenizer.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")

    clip_tokens = clip_tokenizer([caption] * batch_size, truncation=True, padding="max_length", max_length=clip_tokenizer.model_max_length, return_tensors="pt").to(device)
    clip_text_embeddings = clip_model(**clip_tokens).last_hidden_state

    clip_tokens_uncond = clip_tokenizer([negative_caption] * batch_size, truncation=True, padding="max_length", max_length=clip_tokenizer.model_max_length, return_tensors="pt").to(device)
    clip_text_embeddings_uncond = clip_model(**clip_tokens_uncond).last_hidden_state

    del clip_model
    del clip_tokenizer

    return clip_text_embeddings, clip_text_embeddings_uncond

def generateImage(positive_prompt, negative_prompt, image=None):
    #effnet_preprocess = torchvision.transforms.Compose([
    #    torchvision.transforms.Resize(384, interpolation=torchvision.transforms.InterpolationMode.BILINEAR, antialias=True),
    #    torchvision.transforms.CenterCrop(384),
    #    torchvision.transforms.Normalize(
    #        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
    #    )
    #])


    diffuzz = Diffuzz(device=device)

    pretrained_checkpoint = torch.load(checkpoint_stage_b, map_location=device)

    effnet = EfficientNetEncoder(effnet="efficientnet_v2_l").to(device)
    effnet.load_state_dict(pretrained_checkpoint['effnet_state_dict'])
    effnet.eval().requires_grad_(False)

    # - Paella Model as generator -
    generator = Paella(byt5_embd=1024).to(device)
    generator.load_state_dict(pretrained_checkpoint['state_dict'])
    generator.eval().requires_grad_(False)

    del pretrained_checkpoint

    checkpoint = torch.load(checkpoint_stage_c, map_location=device)
    model = Prior(c_in=16, c=1536, c_cond=1024, c_r=64, depth=32, nhead=24).to(device)
    model.load_state_dict(checkpoint['ema_state_dict'])
    model.eval().requires_grad_(False)
    del checkpoint

    batch_size = 4
    prior_timesteps = 60
    prior_cfg = 6
    prior_sampler = "ddpm"

    clip_text_embeddings, clip_text_embeddings_uncond = embed_clip(positive_prompt, negative_prompt, batch_size, device)

    effnet_features_shape = (batch_size, 16, 12, 12)
    effnet_embeddings_uncond = torch.zeros(effnet_features_shape).to(device)
    generator_latent_shape = (batch_size, 128, 128)
    # torch.manual_seed(42)
    with torch.cuda.amp.autocast(dtype=torch.bfloat16), torch.no_grad():
        s = time.time()
        sampled = diffuzz.sample(model, {'c': clip_text_embeddings}, unconditional_inputs={"c": clip_text_embeddings_uncond}, shape=effnet_features_shape,
                                timesteps=prior_timesteps, cfg=prior_cfg, sampler=prior_sampler,
                                t_start=1.0)[-1]
        print(f"Prior Sampling: {time.time() - s}")
        del model

        temperature, cfg, steps =(1.0, 0.6), (2.0, 2.0), 8
        s = time.time()
        sampled_images_original, intermediate = sample(
            generator, {'effnet': sampled,'byt5': clip_text_embeddings}, generator_latent_shape, unconditional_inputs = {'effnet': effnet_embeddings_uncond, 'byt5': clip_text_embeddings_uncond},
            temperature=temperature, cfg=cfg, steps=steps
        )
        print(f"Generator Sampling: {time.time() - s}")

    
    vqmodel = VQModel().to(device)
    vqmodel.load_state_dict(torch.load(checkpoint_stage_a, map_location=device)["state_dict"])
    vqmodel.eval().requires_grad_(False)

    sampled = decode(vqmodel, sampled_images_original)

    del vqmodel

    return sampled




def runModel():
    start_time = time.time()
    directory = "saved_images"
    if not os.path.exists(directory):
        os.makedirs(directory)

    imgs = generateImage("A green landscape with few trees and good weather", "low resolution, low detail, bad quality, blurry")

    # Save each image individually
    for i, img in enumerate(imgs):
        print(img)
        img_to_save = torch.clamp(img, 0, 1)
        img_to_save = img_to_save.permute(1, 2, 0).cpu().numpy()

        # Save the image 
        plt.imsave(os.path.join(directory, f"img_{i}.png"), img_to_save)

    end_time = time.time()

    duration_seconds = end_time - start_time

    minutes = int(duration_seconds // 60)
    seconds = int(duration_seconds % 60)

    

    formatted_time = f"Time taken: {minutes} minutes and {seconds} seconds"

    print(formatted_time)

runModel()