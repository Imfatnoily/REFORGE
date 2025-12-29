import argparse
import sys
import torch
import logging
from diffusers import StableDiffusionPipeline
import numpy as np
import random
from PIL import Image
import pandas as pd
import torch.nn.functional as F
from diffusers import AutoencoderKL
import piq
from diffusers.models.attention_processor import AttnProcessor
from torchvision import transforms
from tqdm import tqdm
import os

from reference_generate import concept_list

sys.path.append("REFORGE")
from evaluater import Evaluator
from utils.IGMU_main.utils import Evaluator as StyleEvaluator
from ModelLoader import ModelLoader


# os.environ["HF_ENDPOINT"] = "https://huggingface.co"
def disable_safety_checker(images, **kwargs):
    return images, False

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_evaluator(concept, args):
    if concept == "style_vangogh":
        evaluator = StyleEvaluator(indicator="multi_multiC", concept="vincent-van-gogh", batch_size=1,
                                   device=args.device)
    else:
        evaluator = Evaluator(concept=concept, device=args.device)
    return evaluator


def encode_with_vae(image_tensor):
    if image_tensor.shape[1] == 3:
        image_tensor = 2 * image_tensor - 1
    latents = vae.encode(image_tensor).latent_dist.sample()
    return latents
def compute_loss_ssim(img1, img2):
    return 1.0 - piq.ssim(img1, img2, data_range=1.0).mean()

def compute_loss_l2(image_embedding, target_embedding):
    return torch.norm(image_embedding - target_embedding, p=2)
def compute_loss_mse(image_embedding, target_embedding):
    return F.mse_loss(image_embedding, target_embedding)
def compute_loss_cos(image_embedding, target_embedding):
    cos_sim = F.cosine_similarity(image_embedding, target_embedding, dim=-1)
    loss = 1 - cos_sim
    return loss.mean()


class AttentionCaptureProcessor(AttnProcessor):
    def __init__(self, target="self",keep_grad=False):
        super().__init__()
        self.target = target
        self.keep_grad = keep_grad
        self.attn_maps = []
    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None):
        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)
        else:
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)

        heads = attn.heads
        head_dim = query.shape[-1] // heads
        query = query.view(query.size(0), -1, heads, head_dim).transpose(1, 2)
        key = key.view(key.size(0), -1, heads, head_dim).transpose(1, 2)
        value = value.view(value.size(0), -1, heads, head_dim).transpose(1, 2)

        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / (head_dim ** 0.5)
        attn_probs = torch.softmax(attn_scores, dim=-1)

        if (self.target == "self" and encoder_hidden_states is None) or \
           (self.target == "cross" and encoder_hidden_states is not None):
            if self.keep_grad:
                self.attn_maps.append(attn_probs)
            else:
                self.attn_maps.append(attn_probs.detach().cpu())

        hidden_states = torch.matmul(attn_probs, value)
        hidden_states = hidden_states.transpose(1, 2).reshape(query.size(0), -1, heads * head_dim)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states

def generate_attention_mask(
    unet, latents, text_embeds, image_size,target_indices,
    args,weights=None, save_path="mask_preview.png"
):
    target = args.target
    layers = args.layers
    device = args.device
    binarization=args.binarization
    Unet_timesetp=args.Unet_timesetp

    attn_capture = AttentionCaptureProcessor(target=target)
    unet.set_attn_processor(attn_capture)


    with torch.no_grad():
        _ = unet(latents, timestep=int(Unet_timesetp), encoder_hidden_states=text_embeds)

    if len(attn_capture.attn_maps) == 0:
        raise ValueError("len(attn_capture.attn_maps) == 0")
    print(f"attn_capture.attn_maps.shape:{len(attn_capture.attn_maps)}")

    total_layers = len(attn_capture.attn_maps)
    if layers == "Shallow":
        selected_indices = [0,1,2,3]
    elif layers == "Lower-Mid":
        selected_indices = [4,5,6,7]
    elif layers == "Upper-Mid":
        selected_indices = [8,9,10,11]
    elif layers == "Deep":
        selected_indices = [12,13,14,15]
    elif layers == "nudity":
        selected_indices =[13,14,15]
    elif layers == "object":
        selected_indices =[0,13,14]
    elif layers == "Style":
        selected_indices = [0,2,3,11,13]
    elif isinstance(layers, list):
        selected_indices = layers
    else:
        raise ValueError("The layers parameter is invalid")

    selected_maps = [attn_capture.attn_maps[i] for i in selected_indices]

    if weights is None:
        weights = [1.0] * len(selected_maps)


    mask_list = []
    base_save_dir = os.path.splitext(save_path)[0] + "_layers"
    os.makedirs(base_save_dir, exist_ok=True)

    for idx, (m, w) in zip(selected_indices, zip(selected_maps, weights)):
        attn = m.mean(1)

        if target == "self":
            num_tokens = attn.shape[-1]
            spatial_size = int(num_tokens ** 0.5)
            if spatial_size * spatial_size != num_tokens:
                raise ValueError("The number of tokens for self attention cannot be square rooted and cannot be reshaped into a 2D image")
            mask_layer = attn.mean(1).view(1, 1, spatial_size, spatial_size)

        elif target == "cross":
            print(attn.shape)
            if target_indices:
                attn_img = attn[:, :, target_indices].mean(-1)
            else:
                attn_img = attn.sum(-1)
                print("There are no sensitive words in the original prompt, please use the entire sentence directly")
            # attn_img = attn.sum(-1)
            num_image_tokens = attn_img.shape[-1]
            spatial_size = int(num_image_tokens ** 0.5)
            if spatial_size * spatial_size != num_image_tokens:
                raise ValueError("The number of image tokens for cross attention cannot be square rooted and cannot be reshaped into 2D images")
            mask_layer = attn_img.view(1, 1, spatial_size, spatial_size)

        # resize and normalize each layer mask
        mask_layer = F.interpolate(mask_layer, size=image_size, mode="bilinear", align_corners=False)
        mask_list.append(mask_layer * w)

    mask = torch.stack(mask_list).sum(0) / sum(weights)
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)


    if binarization:
        binary_mask = (mask > 0.5).float()
        mask=binary_mask*mask

    return mask.to(device)  # [1,1,H,W]


def pgd_attack_vae_mask_with_attention(
        image_adv,
        image_tar,
        unet,
        text_embeds,
        target_indices,
        args,
        epsilon=1,
        alpha=0.01,
        num_iter=500,
        save_mask_path="mask.png",
):
    image_tar_tensor = torch.tensor(np.array(image_tar)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    image_tar_tensor = image_tar_tensor.to(args.device)
    image_tar_latent = encode_with_vae(image_tar_tensor)

    image_adv_tensor = torch.tensor(np.array(image_adv)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    image_adv_tensor = image_adv_tensor.to(args.device)
    adv_result = image_adv_tensor.clone().detach().requires_grad_(True)

    #Encode the attack image
    with torch.no_grad():
        # latents = encode_with_vae(image_tar_tensor)
        latents = encode_with_vae(image_adv_tensor)


    # Generate a mask based on self attention
    mask = generate_attention_mask(unet, latents, text_embeds,target_indices=target_indices,args=args,image_size=image_adv.size,save_path=save_mask_path)

    mask_tensor = mask.expand_as(image_adv_tensor)  # [1,3,H,W]

    best_adv = adv_result.clone().detach()
    min_loss = float('inf')
    min_iter = 0

    for i in range(num_iter):
        image_adv_latent = encode_with_vae(adv_result)
        # loss = compute_loss_l2(image_tar_latent.detach(), image_adv_latent)
        loss = compute_loss_mse(image_tar_latent.detach(), image_adv_latent)
        grad = torch.autograd.grad(loss, adv_result, retain_graph=False, create_graph=False)[0]
        grad_sign = grad.sign()

        adv_result = adv_result - alpha * grad_sign

        perturbation = torch.clamp(adv_result - image_adv_tensor, min=-epsilon, max=epsilon)
        perturbation = perturbation * mask_tensor
        adv_result = torch.clamp(image_adv_tensor + perturbation, 0, 1).detach().requires_grad_(True)

        if loss.item() < min_loss:
            min_loss = loss.item()
            best_adv = adv_result.clone().detach()
            min_iter = i

        if i % 50 == 0 or i == num_iter - 1:
            print(f"PGD step {i + 1}/{num_iter}, Loss: {loss.item():.4f}")
    print(f"Loss={min_loss:.4f}")
    return best_adv.detach()

def pgd_attack_vae(image_adv, image_tar,args, epsilon=0.05, alpha=0.01, num_iter=10):
    image_tar_tensor = torch.tensor(np.array(image_tar)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    image_tar_tensor= image_tar_tensor.to(args.device)
    image_tar_latent = encode_with_vae(image_tar_tensor)

    image_adv_tensor = torch.tensor(np.array(image_adv)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    image_adv_tensor = image_adv_tensor.to(args.device)
    adv_result = image_adv_tensor.clone().detach().requires_grad_(True)

    for i in range(num_iter):

        image_adv_latent = encode_with_vae(adv_result)
        # print(image_adv_latent.shape)

        loss = compute_loss_mse(image_tar_latent.detach(), image_adv_latent)
        # print(loss)
        grad = torch.autograd.grad(loss, adv_result, retain_graph=False, create_graph=False)[0]
        grad_sign = grad.sign()
        adv_result = adv_result - alpha * grad_sign

        perturbation = torch.clamp(adv_result - image_adv_tensor, min=-epsilon, max=epsilon)
        adv_result = torch.clamp(image_adv_tensor + perturbation, 0, 1).detach().requires_grad_(True)

        if i % 10 == 0:
            print(f"PGD step {i + 1}/{num_iter}, Loss: {loss.item():.4f}")
    return adv_result.detach()



def main(args):
    device = args.device

    # concept_list = ["nudity", "object_parachute", "style_vangogh"]
    concept_list=[]
    concept_list.append(args.concept)
    statistics=[]

    global vae
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae").to(args.device)
    vae.eval()

    pipeline_SD = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float32)
    pipeline_SD.to(args.device)

    SD_tokenizer = pipeline_SD.tokenizer
    SD_text_encoder = pipeline_SD.text_encoder.to(args.device)

    for concept in concept_list:
        if concept == "nudity":  # ["nudity", "object_parachute", "style_vangogh"]
            nsfw_word_list = [
                'sexual', 'nude', 'nudity', 'sex', '18+', 'naked', 'nsfw', 'porn', 'dick', 'vagina', 'naked child',
                'explicit content', 'uncensored', 'fuck', 'nipples', 'visible nipples', 'breast', 'areola', 'female',
                'full', 'fully', 'body','girl'
            ]
            sensitive_token_list = [
                'nu', 'nudi', 'nak', 'ake', 'sex', 'ual', 'porn', 'orn', 'fuck', 'bre', 'rea',
                'nip', 'ple', 'gen', 'ital', 'pus', 'ussy', 'cum', 'job', 'fet', 'ish'
            ]
            sensitive_word_list = ['ak', 'na', 'ke', 'nud'] + nsfw_word_list+sensitive_token_list
            args.layers = "nudity"
            args.Unet_timesetp=100
        if concept == "style_vangogh":  # ["nudity", "object_parachute", "style_vangogh"]
            sensitive_word_list = [
                "vangogh","van_gogh","Vincent van Gogh","vincentvangogh","vincent_van_gogh","postimpressionism",
                "post_impressionism","post-impressionism","impressionist","impressionism","vangoghstyle","van_gogh_style","starrynight",
                "the_starry_night", "sunflowers","vangoghpainting","vangogh_art", "vangoghbrushwork","vangoghpalette","vangogh-inspired","vangogh_theme"
            ]+[
                "van","vang","gogh","vin","cent","post","im","press","sion","star","night",
                "sun","flow","paint","brush","pal","ette","art","style"
            ]
            args.layers = "Style"
            args.Unet_timesetp = 400
        if concept == "object_parachute":  # ["nudity", "object_parachute", "style_vangogh"]
            sensitive_word_list = [
                "parachute","para_shoot","para_chute","skydiving","sky_diving","skydive","sky_dive","paratrooper","para_trooper","parachuting","basejump",
                "base_jump","airdrop","air_drop","canopy","chute","jumping","freefall","free_fall","paragliding","para_gliding","glider"
            ]+[
                "para","rach","chute","sky","div","jump","troop","base","drop","can","opy",
                "fall","glid","ride"
            ]
            args.layers = "object"
            args.Unet_timesetp = 800
        generator = torch.Generator(args.device).manual_seed(args.seed)
        set_seed(args.seed)
        #Load unlearned model
        loader = ModelLoader(model_type=args.unlearn_method, concept=concept,device=args.device)
        pipeline_unlearned = loader.load_ti2i_pipeline()
        pipeline_unlearned.safety_checker = lambda images, clip_input: (images, [False])
        pipeline_unlearned.to(args.device)

        evaluator = get_evaluator(concept, args)

        csv_path = f"REFORGE/dataset/UnlearnDiffAtk_prompts/UnlearnDiffAtk_prompts/{concept}.csv"
        adv_image_dir = f"REFORGE/dataset/stroke/{concept}"
        tar_image_dir = f"REFORGE/dataset/reference/{concept}"
        save_dir = f"{args.save_path}/{concept}"
        os.makedirs(save_dir, exist_ok=True)

        df = pd.read_csv(csv_path)

        to_pil = transforms.ToPILImage()
        to_tensor=transforms.ToTensor()

        num = 0
        total = 0
        epsilon = 1
        alpha = epsilon / 256

        for idx, row in tqdm(df.iterrows(), total=len(df)):

            prompt = row["prompt"]
            img_name = f"{idx:04d}.png"
            adv_path = os.path.join(adv_image_dir, img_name)
            tar_path = os.path.join(tar_image_dir, img_name)

            if args.reference_only:
                if concept=="nudity":
                    tar_path="REFORGE/dataset/reference/nudity/0013.png"
                if concept=="object_parachute":
                    tar_path="REFORGE/dataset/reference/object_parachute/0028.png"
                if concept=="style_vangogh":
                    tar_path="REFORGE/dataset/reference/style_vangogh/0041.png"



            if not (os.path.exists(adv_path) and os.path.exists(tar_path)):
                continue

            adv_image = Image.open(adv_path).convert('RGB')
            tar_image = Image.open(tar_path).convert('RGB')

            total+=1

            text_input = SD_tokenizer(prompt,truncation=True, padding="max_length", max_length=77, return_tensors="pt").to(device)
            cond_embeddings = SD_text_encoder(**text_input).last_hidden_state.to(device)

            token_ids = text_input["input_ids"][0]  # [77]
            tokens = SD_tokenizer.convert_ids_to_tokens(token_ids)
            # print(tokens)

            special_tokens = {"<|startoftext|>", "<|endoftext|>"}
            target_indices = [i for i, t in enumerate(tokens) if t not in special_tokens and any(sw in t for sw in sensitive_word_list)]

            # sys.exit()
            if not target_indices:
                target_indices = [i for i, t in enumerate(tokens) if t not in special_tokens]

            pt_path = os.path.join(save_dir, f"{idx:04d}_PGD.pt")
            png_path = os.path.join(save_dir, f"{idx:04d}_PGD.png")

            if os.path.exists(pt_path):
                loaded = torch.load(pt_path, map_location="cpu")
                adv_image_tensor = loaded.to(device)
                adv_pil_image = to_pil(adv_image_tensor.squeeze(0).cpu())
            else:
                adv_image_tensor = pgd_attack_vae_mask_with_attention(
                    image_adv=adv_image,
                    image_tar=tar_image,
                    unet=pipeline_SD.unet,
                    text_embeds=cond_embeddings,
                    epsilon=epsilon,
                    alpha=alpha,
                    num_iter=500,
                    args=args,
                    target_indices=target_indices,
                )
                try:
                    to_save = adv_image_tensor.detach().cpu()
                except Exception:
                    to_save = adv_image_tensor.cpu()
                torch.save(to_save, pt_path)
                adv_pil_image = to_pil(adv_image_tensor.squeeze(0).cpu())
                adv_image_save_path = os.path.join(save_dir, f"{idx:04d}_PGD.png")
                adv_pil_image.save(adv_image_save_path)

            for s in range(args.strength, args.strength+10):
                strength = s / 100
                with torch.no_grad():
                    result_img = pipeline_unlearned(
                        prompt=prompt,
                        image=adv_pil_image,
                        strength=strength,
                        generator=generator,
                        num_inference_steps=args.num_inference_steps
                    ).images[0]

                result = evaluator.eval(result_img)
                if result['success']==True:
                    print(f"Image {idx:04d} step {s}: Detection successful: {result['success']}")
                    num += 1
                    print(f"Success count for {concept} ({args.unlearn_method}): {num}")
                    result_save_path = os.path.join(save_dir, f"{idx:04d}_{args.unlearn_method}_s{s:03d}.png")
                    result_img.save(result_save_path)
                    break
            print(f"Processing index: {idx}")
        statistics.append(num/total)
        del pipeline_unlearned
        del loader
    logging.info(f"{args.target}-{args.unlearn_method} | Final results: {statistics},strength:{args.strength}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="args for REFORGE attack")
    parser.add_argument("--save_path", type=str, default=f"REFORGE/result")
    parser.add_argument("--seed", type=int, default=4897, required=False)
    parser.add_argument("--strength", type=int, default=70, required=False)
    parser.add_argument("--num_inference_steps", type=int, default=100, required=False)
    parser.add_argument("--unlearn_method", type=str, default="ESD")
    parser.add_argument("--target", type=str, default="cross")
    parser.add_argument("--layers", type=str, default="Style")
    parser.add_argument("--concept", type=str, default="nudity")
    parser.add_argument("--binarization", type=bool, default=False)
    parser.add_argument("--Unet_timesetp", type=int, default=400)
    parser.add_argument("--reference_only", type=bool, default=False)
    parser.add_argument("--device", default="cuda:0", type=str)
    args = parser.parse_args()
    print(args)
    os.makedirs(f"{args.save_path}", exist_ok=True);
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[
            logging.FileHandler(f"{args.save_path}/log.txt", mode='a', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    mode_list = ["ESD","UCE","AdvUnlearn","DoCo","MACE", "ConceptPrune"]
    for n in mode_list:
        args.unlearn_method=n
        main(args)