import os
import math
import gradio as gr
import numpy as np
import torch
import safetensors.torch as sf
import random
import db_examples
import sys
import platform 
from PIL import Image
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler, EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler
from diffusers.models.attention_processor import AttnProcessor2_0
from transformers import CLIPTextModel, CLIPTokenizer
from briarmbg import BriaRMBG
from enum import Enum
from torch.hub import download_url_to_file

MAX_SEED = np.iinfo(np.int32).max
def open_folder():
    open_folder_path = os.path.abspath("outputs")
    if platform.system() == "Windows":
        os.startfile(open_folder_path)
    elif platform.system() == "Linux":
        os.system(f'xdg-open "{open_folder_path}"')


# 'stablediffusionapi/realistic-vision-v51'
# 'runwayml/stable-diffusion-v1-5'
sd15_name = 'stablediffusionapi/realistic-vision-v51'
tokenizer = CLIPTokenizer.from_pretrained(sd15_name, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(sd15_name, subfolder="text_encoder")
vae = AutoencoderKL.from_pretrained(sd15_name, subfolder="vae")
unet = UNet2DConditionModel.from_pretrained(sd15_name, subfolder="unet")
rmbg = BriaRMBG.from_pretrained("briaai/RMBG-1.4")

# Change UNet

with torch.no_grad():
    new_conv_in = torch.nn.Conv2d(8, unet.conv_in.out_channels, unet.conv_in.kernel_size, unet.conv_in.stride, unet.conv_in.padding)
    new_conv_in.weight.zero_()
    new_conv_in.weight[:, :4, :, :].copy_(unet.conv_in.weight)
    new_conv_in.bias = unet.conv_in.bias
    unet.conv_in = new_conv_in

unet_original_forward = unet.forward


def hooked_unet_forward(sample, timestep, encoder_hidden_states, **kwargs):
    c_concat = kwargs['cross_attention_kwargs']['concat_conds'].to(sample)
    c_concat = torch.cat([c_concat] * (sample.shape[0] // c_concat.shape[0]), dim=0)
    new_sample = torch.cat([sample, c_concat], dim=1)
    kwargs['cross_attention_kwargs'] = {}
    return unet_original_forward(new_sample, timestep, encoder_hidden_states, **kwargs)


unet.forward = hooked_unet_forward

# Load

model_path = './models/iclight_sd15_fc.safetensors'

if not os.path.exists(model_path):
    download_url_to_file(url='https://huggingface.co/lllyasviel/ic-light/resolve/main/iclight_sd15_fc.safetensors', dst=model_path)

sd_offset = sf.load_file(model_path)
sd_origin = unet.state_dict()
keys = sd_origin.keys()
sd_merged = {k: sd_origin[k] + sd_offset[k] for k in sd_origin.keys()}
unet.load_state_dict(sd_merged, strict=True)
del sd_offset, sd_origin, sd_merged, keys

# Device

device = torch.device('cuda')
text_encoder = text_encoder.to(device=device, dtype=torch.float16)
vae = vae.to(device=device, dtype=torch.bfloat16)
unet = unet.to(device=device, dtype=torch.float16)
rmbg = rmbg.to(device=device, dtype=torch.float32)

# SDP

unet.set_attn_processor(AttnProcessor2_0())
vae.set_attn_processor(AttnProcessor2_0())

# Samplers

ddim_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
)

euler_a_scheduler = EulerAncestralDiscreteScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    steps_offset=1
)

dpmpp_2m_sde_karras_scheduler = DPMSolverMultistepScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    algorithm_type="sde-dpmsolver++",
    use_karras_sigmas=True,
    steps_offset=1
)

# Pipelines

t2i_pipe = StableDiffusionPipeline(
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    unet=unet,
    scheduler=dpmpp_2m_sde_karras_scheduler,
    safety_checker=None,
    requires_safety_checker=False,
    feature_extractor=None,
    image_encoder=None
)

i2i_pipe = StableDiffusionImg2ImgPipeline(
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    unet=unet,
    scheduler=dpmpp_2m_sde_karras_scheduler,
    safety_checker=None,
    requires_safety_checker=False,
    feature_extractor=None,
    image_encoder=None
)


@torch.inference_mode()
def encode_prompt_inner(txt: str):
    max_length = tokenizer.model_max_length
    chunk_length = tokenizer.model_max_length - 2
    id_start = tokenizer.bos_token_id
    id_end = tokenizer.eos_token_id
    id_pad = id_end

    def pad(x, p, i):
        return x[:i] if len(x) >= i else x + [p] * (i - len(x))

    tokens = tokenizer(txt, truncation=False, add_special_tokens=False)["input_ids"]
    chunks = [[id_start] + tokens[i: i + chunk_length] + [id_end] for i in range(0, len(tokens), chunk_length)]
    chunks = [pad(ck, id_pad, max_length) for ck in chunks]

    token_ids = torch.tensor(chunks).to(device=device, dtype=torch.int64)
    conds = text_encoder(token_ids).last_hidden_state

    return conds


@torch.inference_mode()
def encode_prompt_pair(positive_prompt, negative_prompt):
    c = encode_prompt_inner(positive_prompt)
    uc = encode_prompt_inner(negative_prompt)

    c_len = float(len(c))
    uc_len = float(len(uc))
    max_count = max(c_len, uc_len)
    c_repeat = int(math.ceil(max_count / c_len))
    uc_repeat = int(math.ceil(max_count / uc_len))
    max_chunk = max(len(c), len(uc))

    c = torch.cat([c] * c_repeat, dim=0)[:max_chunk]
    uc = torch.cat([uc] * uc_repeat, dim=0)[:max_chunk]

    c = torch.cat([p[None, ...] for p in c], dim=1)
    uc = torch.cat([p[None, ...] for p in uc], dim=1)

    return c, uc


@torch.inference_mode()
def pytorch2numpy(imgs, quant=True):
    results = []
    for x in imgs:
        y = x.movedim(0, -1)

        if quant:
            y = y * 127.5 + 127.5
            y = y.detach().float().cpu().numpy().clip(0, 255).astype(np.uint8)
        else:
            y = y * 0.5 + 0.5
            y = y.detach().float().cpu().numpy().clip(0, 1).astype(np.float32)

        results.append(y)
    return results


@torch.inference_mode()
def numpy2pytorch(imgs):
    h = torch.from_numpy(np.stack(imgs, axis=0)).float() / 127.0 - 1.0  # so that 127 must be strictly 0.0
    h = h.movedim(-1, 1)
    return h


def resize_and_center_crop(image, target_width, target_height):
    pil_image = Image.fromarray(image)
    original_width, original_height = pil_image.size
    scale_factor = max(target_width / original_width, target_height / original_height)
    resized_width = int(round(original_width * scale_factor))
    resized_height = int(round(original_height * scale_factor))
    resized_image = pil_image.resize((resized_width, resized_height), Image.LANCZOS)
    left = (resized_width - target_width) / 2
    top = (resized_height - target_height) / 2
    right = (resized_width + target_width) / 2
    bottom = (resized_height + target_height) / 2
    cropped_image = resized_image.crop((left, top, right, bottom))
    return np.array(cropped_image)


def resize_without_crop(image, target_width, target_height):
    pil_image = Image.fromarray(image)
    resized_image = pil_image.resize((target_width, target_height), Image.LANCZOS)
    return np.array(resized_image)


@torch.inference_mode()
def run_rmbg(img, sigma=0.0):
    H, W, C = img.shape
    assert C == 3
    k = (256.0 / float(H * W)) ** 0.5
    feed = resize_without_crop(img, int(64 * round(W * k)), int(64 * round(H * k)))
    feed = numpy2pytorch([feed]).to(device=device, dtype=torch.float32)
    alpha = rmbg(feed)[0][0]
    alpha = torch.nn.functional.interpolate(alpha, size=(H, W), mode="bilinear")
    alpha = alpha.movedim(1, -1)[0]
    alpha = alpha.detach().float().cpu().numpy().clip(0, 1)
    result = 127 + (img.astype(np.float32) - 127 + sigma) * alpha
    return result.clip(0, 255).astype(np.uint8), alpha


@torch.inference_mode()
def process(input_fg, prompt, image_width, image_height, num_samples, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, lowres_denoise, bg_source):
    bg_source = BGSource(bg_source)
    input_bg = None

    if bg_source == BGSource.NONE:
        pass
    elif bg_source == BGSource.LEFT:
        gradient = np.linspace(255, 0, image_width)
        image = np.tile(gradient, (image_height, 1))
        input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
    elif bg_source == BGSource.RIGHT:
        gradient = np.linspace(0, 255, image_width)
        image = np.tile(gradient, (image_height, 1))
        input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
    elif bg_source == BGSource.TOP:
        gradient = np.linspace(255, 0, image_height)[:, None]
        image = np.tile(gradient, (1, image_width))
        input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
    elif bg_source == BGSource.BOTTOM:
        gradient = np.linspace(0, 255, image_height)[:, None]
        image = np.tile(gradient, (1, image_width))
        input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
    else:
        raise 'Wrong initial latent!'

    rng = torch.Generator(device=device).manual_seed(int(seed))

    fg = resize_and_center_crop(input_fg, image_width, image_height)

    concat_conds = numpy2pytorch([fg]).to(device=vae.device, dtype=vae.dtype)
    concat_conds = vae.encode(concat_conds).latent_dist.mode() * vae.config.scaling_factor

    conds, unconds = encode_prompt_pair(positive_prompt=prompt + ', ' + a_prompt, negative_prompt=n_prompt)

    if input_bg is None:
        latents = t2i_pipe(
            prompt_embeds=conds,
            negative_prompt_embeds=unconds,
            width=image_width,
            height=image_height,
            num_inference_steps=steps,
            num_images_per_prompt=num_samples,
            generator=rng,
            output_type='latent',
            guidance_scale=cfg,
            cross_attention_kwargs={'concat_conds': concat_conds},
        ).images.to(vae.dtype) / vae.config.scaling_factor
    else:
        bg = resize_and_center_crop(input_bg, image_width, image_height)
        bg_latent = numpy2pytorch([bg]).to(device=vae.device, dtype=vae.dtype)
        bg_latent = vae.encode(bg_latent).latent_dist.mode() * vae.config.scaling_factor
        latents = i2i_pipe(
            image=bg_latent,
            strength=lowres_denoise,
            prompt_embeds=conds,
            negative_prompt_embeds=unconds,
            width=image_width,
            height=image_height,
            num_inference_steps=int(round(steps / lowres_denoise)),
            num_images_per_prompt=num_samples,
            generator=rng,
            output_type='latent',
            guidance_scale=cfg,
            cross_attention_kwargs={'concat_conds': concat_conds},
        ).images.to(vae.dtype) / vae.config.scaling_factor

    pixels = vae.decode(latents).sample
    pixels = pytorch2numpy(pixels)
    pixels = [resize_without_crop(
        image=p,
        target_width=int(round(image_width * highres_scale / 64.0) * 64),
        target_height=int(round(image_height * highres_scale / 64.0) * 64))
    for p in pixels]

    pixels = numpy2pytorch(pixels).to(device=vae.device, dtype=vae.dtype)
    latents = vae.encode(pixels).latent_dist.mode() * vae.config.scaling_factor
    latents = latents.to(device=unet.device, dtype=unet.dtype)

    image_height, image_width = latents.shape[2] * 8, latents.shape[3] * 8

    fg = resize_and_center_crop(input_fg, image_width, image_height)
    concat_conds = numpy2pytorch([fg]).to(device=vae.device, dtype=vae.dtype)
    concat_conds = vae.encode(concat_conds).latent_dist.mode() * vae.config.scaling_factor

    latents = i2i_pipe(
        image=latents,
        strength=highres_denoise,
        prompt_embeds=conds,
        negative_prompt_embeds=unconds,
        width=image_width,
        height=image_height,
        num_inference_steps=int(round(steps / highres_denoise)),
        num_images_per_prompt=num_samples,
        generator=rng,
        output_type='latent',
        guidance_scale=cfg,
        cross_attention_kwargs={'concat_conds': concat_conds},
    ).images.to(vae.dtype) / vae.config.scaling_factor

    pixels = vae.decode(latents).sample

    return pytorch2numpy(pixels)


@torch.inference_mode()
def process_relight(input_fg, prompt, image_width, image_height, num_samples, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, lowres_denoise, bg_source, randomize_seed):
    input_fg, matting = run_rmbg(input_fg)
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    results = process(input_fg, prompt, image_width, image_height, num_samples, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, lowres_denoise, bg_source)
    
    # Generate outputs folder if it doesn't exist
    os.makedirs('outputs', exist_ok=True)
    
    # Find the latest available number for saving images
    existing_files = os.listdir('outputs')
    existing_numbers = [int(file.split('.')[0].split('_')[1]) for file in existing_files if file.endswith('.png')]
    latest_number = max(existing_numbers) if existing_numbers else 0
    
    # Save each generated image with the next available number
    for i, result in enumerate(results):
        image_number = latest_number + i + 1
        filename = f'img_{image_number:05d}.png'
        filepath = os.path.join('outputs', filename)
        Image.fromarray(result).save(filepath)
    
    return input_fg, results, seed  # Return the seed along with input_fg and results


quick_prompts = [
'STARTERS',
'sunshine from window',
'neon light, city',
'sunset over sea',
'golden time',
'sci-fi RGB glowing, cyberpunk',
'natural lighting',
'warm atmosphere, at home, bedroom',
'magic lit',
'evil, gothic, Yharnam',
'light and shadow',
'shadow from window',
'soft studio lighting',
'home atmosphere, cozy bedroom illumination',
'neon, Wong Kar-wai, warm',
'Moonlight through trees',
'Starry night sky',
'Northern lights, aurora borealis',
'Candlelit dinner',
'Thunderstorm with lightning',
'Twilight, after sunset',
'Campfire glow',
'Underwater light beams',
'Fireworks display',
'Lantern-lit street',
'Sunbeams in a forest',
'Streetlights in rain',
'Reflections on water',
'Misty morning sunrise',
'Foggy street with lamps',
'Bioluminescent sea',
'Flickering TV light',
'Cityscape at night',
'Light through stained glass',
'Candlelit library',
'Red carpet event flash',
'Golden hour in the mountains',
'Halloween pumpkin light',
'Festival fairy lights',
'Soft morning sunlight',
'Moonlight on snow',
'Light from fireplace',
'City skyline at dusk',
'Spotlight on stage',
'Soft glow from laptop screen',
'',
'BRIGHT AMBIENT',
'Golden sunrise over a meadow',
'Bright afternoon sunlight in a park',
'Sunlit botanical garden',
'Spring morning with cherry blossoms',
'Ocean waves under bright sunlight',
'Countryside with rolling hills and sunshine',
'Open field with wildflowers in daylight',
'Rooftop terrace with midday sun',
'Bright beach day with clear skies',
'City park at noon',
'Sunlight streaming through a glass ceiling',
'Mountain peak with bright sky',
'Sunny courtyard in a Mediterranean villa',
'Greenhouse with natural light',
'Lakeside with clear blue skies',
'Sunlit tropical rainforest',
'Hiking trail with midday sun',
'Bright sunflower field',
'Sunny vineyard',
'White sandy beach under the sun',
'Sunlight through a canopy of trees',
'Bright snowy landscape',
'Summer fairground in daylight',
'Sun-drenched patio',
'Daylight in a bustling market',
'Sunlit waterfall in a forest',
'Countryside cottage with morning sun',
'Bright city square',
'Daytime carnival with balloons and sunshine',
'Sunlit tulip field in the spring',
'',
'',
'DYNAMIC DRAMATIC',
'Volcanic eruption glow',
'Underwater cave with bioluminescence',
'Stormy sea with lightning',
'Desert sunset with sandstorm',
'Firefly-lit forest',
'Eclipse shadow',
'Gothic cathedral with moonlight',
'Rainforest with dappled sunlight',
'City skyline with aurora australis',
'Floodlights at a concert',
'Dramatic shadows from blinds',
'Snowstorm with headlights',
'Sunlight through a stormy cloud',
'Spotlights in a dark alley',
'Laser light show',
'Abandoned building with broken windows',
'Carnival with neon lights',
'Opera house with stage lights',
'Under a solar eclipse',
'Sunrise over a misty valley',
'Stormy beach with lighthouse beam',
'Haunted house with candle flicker',
'Desert night with star trails',
'Thunderstorm over a cityscape',
'Sunlight through ancient ruins',
'Industrial area with steam and lights',
'Underwater ruins with light shafts',
'Moonlit graveyard',
'Sunset behind a mountain silhouette',
'Train station with fog and headlights',
'',
'NATURAL',
'Morning dew with sunlight on grass',
'Sunset over rolling hills',
'Sunlight filtering through autumn leaves',
'Golden hour on a lake',
'Sunrise over a foggy meadow',
'Mountain range at dawn',
'Sunlight through bamboo forest',
'Cave opening with sunlight',
'Sunset over desert dunes',
'Sunlight on coral reef underwater',
'Morning light in an orchard',
'Sunlight through a canyon',
'Sunset over lavender fields',
'Sunlight through tall grass',
'Treetop canopy with scattered sunlight',
'Sunrise over a snowy landscape',
'Sunlight in a rustic barn',
'Sunlight on a riverbank',
'Sunlight through rain on a window',
'Sunrise over a fishing village',
'Sunlight in a flower garden',
'Sunset over a vineyard',
'Morning light on a mountain trail',
'Sunlight through coastal cliffs',
'Sunset over a wetland',
'Sunlight in a forest glade',
'Sunlight on a pebble beach',
'Sunrise in a mountain valley',
'Sunlight through pine forest',
'Sunset over a rice paddy field',
'',
'PROFESSIONAL MODEL PHOTOSHOOT BACKGROUND',
'Golden hour in a modern cityscape',
'Sunlight through a minimalist window',
'Soft natural light in a white studio',
'Backlit silhouette at sunset',                               
'Soft morning light in a chic apartment',
'Bright daylight in an industrial loft',
'Natural light on a rooftop garden',
'Sunset over a stylish urban rooftop',
'Sunlight through sheer curtains in a modern living room',
'Natural light in a trendy cafe',
'Bright afternoon in a botanical garden',
'Sunlit alley with cobblestones',
'Bright and airy greenhouse',
'Sunlight through a floor-to-ceiling window in a studio',
'Sunset on a terrace with urban views',
'Bright natural light in a spacious warehouse',
'Morning light in a sophisticated library',
'Sunlight on a marble staircase',
'Bright daylight in an art gallery',
'Golden hour in a scenic vineyard',
'Sunlit patio with modern furniture',
'Natural light in a high-rise office',
'Sunset over a luxury beach resort',
'Bright daylight in a stylish courtyard',
'Morning light on a modern balcony',
'Sunlight in a loft with exposed brick walls',
'Soft natural light in a cozy bedroom',
'Sunlit hallway with large windows',
'Bright and open modern kitchen',
'Golden hour in a chic urban park',
'Soft natural light in a bohemian-style room',
'Sunset over a chic rooftop lounge',
'Golden hour in a serene garden',
'Morning light in a Parisian-style cafe',
'Sunlight through lace curtains in a vintage bedroom',
'Bright daylight in a minimalist studio',
'Sunset on a sandy beach with gentle waves',
'Natural light in a rustic barn with hay bales',
'Soft morning light in a Victorian greenhouse',
'Sunlit floral archway in a botanical garden',
'Golden hour in a lavender field',
'Bright daylight in an airy loft with exposed beams',
'Sunset on a modern balcony with city views',
'Soft natural light in a marble-floored atrium',
'Morning light in a chic, white-walled living room',
'Golden hour in a sunflower field',
'Sunlit cobblestone street in an old town',
'Bright daylight in a stylish courtyard with a fountain',
'Soft light in a luxurious dressing room with mirrors',
'Sunset over a serene lake with a pier',
'Morning light in a quaint country kitchen',
'Golden hour in a vineyard with rolling hills',
'Sunlight through sheer curtains in a modern bedroom',
'Bright natural light in a high-rise office with panoramic views',
'Sunset in a chic urban park with modern sculptures',
'Morning light in a spacious art studio',
'Soft natural light in a cozy reading nook',
'Golden hour in a rose garden',
'Sunlit walkway with blooming bougainvillea',
'Bright daylight in an elegant ballroom with chandeliers',
'Golden hour on a rugged coastline',
'Morning light in a modern industrial loft',
'Sunset over a city skyline',
'Soft natural light in a minimalist apartment',
'Bright daylight in an urban graffiti alley',
'Sunlight through large warehouse windows',
'Sunset on a rooftop with a skyline view',
'Natural light in a vintage barbershop',
'Golden hour in a desert landscape',
'Morning light in a sleek, modern office',
'Sunlight through a forest canopy',
'Bright daylight in a concrete skate park',
'Sunset over a marina with yachts',
'Natural light in a rustic cabin',
'Golden hour in an urban park with sculptures',
'Morning light in a gym with large windows',
'Sunlight through floor-to-ceiling windows in a modern condo',
'Bright daylight in a vintage car garage',
'Soft natural light in a stylish library',
'Golden hour on a mountain trail',
'Sunset on a pier with industrial elements',
'Morning light in a classic coffee shop',
'Sunlight streaming through blinds in a chic apartment',
'Bright daylight in a modern art gallery',
'Golden hour in a field with tall grass',
'Morning light in a sleek, modern kitchen',
'Sunset over an old town square',
'Natural light in a high-rise loft with city views',
'Sunlight on an urban rooftop garden',
'Golden hour in an industrial chic restaurant'
]
quick_prompts = [[x] for x in quick_prompts]


quick_subjects = [
    'beautiful woman, detailed face',
    'handsome man, detailed face',
]
quick_subjects = [[x] for x in quick_subjects]


class BGSource(Enum):
    NONE = "None"
    LEFT = "Left Light"
    RIGHT = "Right Light"
    TOP = "Top Light"
    BOTTOM = "Bottom Light"


block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown(""" IC-Light (Relighting with Foreground Condition) - V4 - This is improved version of publicly released Gradio demo
        ### 如需帮助请联系匹夫微信：AI-pifu """)
    with gr.Row():
        with gr.Column():
            with gr.Row():
                input_fg = gr.Image(source='upload', type="numpy", label="Image", height=480)
                output_bg = gr.Image(type="numpy", label="Preprocessed Foreground", height=480)
            prompt = gr.Textbox(label="Prompt")
            bg_source = gr.Radio(choices=[e.value for e in BGSource],
                                 value=BGSource.NONE.value,
                                 label="Lighting Preference (Initial Latent)", type='value')
            example_quick_subjects = gr.Dataset(samples=quick_subjects, label='Subject Quick List', samples_per_page=1000, components=[prompt])
            example_quick_prompts = gr.Dataset(samples=quick_prompts, label='Lighting Quick List', samples_per_page=1000, components=[prompt])

        with gr.Column():
            result_gallery = gr.Gallery(height=768, object_fit='contain', label='Outputs')
            relight_button = gr.Button(value="Relight")

            with gr.Group():
                with gr.Row():
                    num_samples = gr.Slider(label="Batch Size", minimum=1, maximum=12, value=1, step=1)
                    seed = gr.Number(label="Seed", value=12345, precision=0)
                    randomize_seed = gr.Checkbox(label="Randomize Seed", value=True)  # Add this line

                with gr.Row():
                    image_width = gr.Slider(label="Image Width", minimum=256, maximum=1024, value=512, step=64)
                    image_height = gr.Slider(label="Image Height", minimum=256, maximum=1024, value=768, step=64)

            with gr.Accordion("Advanced options", open=False):
                steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=25, step=1)
                cfg = gr.Slider(label="CFG Scale", minimum=1.0, maximum=32.0, value=2, step=0.01)
                lowres_denoise = gr.Slider(label="Lowres Denoise (for initial latent)", minimum=0.1, maximum=1.0, value=0.9, step=0.01)
                highres_scale = gr.Slider(label="Highres Scale", minimum=1.0, maximum=3.0, value=1.5, step=0.01)
                highres_denoise = gr.Slider(label="Highres Denoise", minimum=0.1, maximum=1.0, value=0.5, step=0.01)
                a_prompt = gr.Textbox(label="Added Prompt", value='best quality')
                n_prompt = gr.Textbox(label="Negative Prompt", value='lowres, bad anatomy, bad hands, cropped, worst quality')
            btn_open_outputs = gr.Button("Open Outputs Folder")
            btn_open_outputs.click(fn=open_folder)

    ips = [input_fg, prompt, image_width, image_height, num_samples, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, lowres_denoise, bg_source, randomize_seed]
    relight_button.click(fn=process_relight, inputs=ips, outputs=[output_bg, result_gallery, seed])
    example_quick_prompts.click(lambda x, y: ', '.join(y.split(', ')[:2] + [x[0]]), inputs=[example_quick_prompts, prompt], outputs=prompt, show_progress=False, queue=False)
    example_quick_subjects.click(lambda x: x[0], inputs=example_quick_subjects, outputs=prompt, show_progress=False, queue=False)


block.launch(inbrowser=True)
