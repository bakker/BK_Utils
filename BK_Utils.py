import datetime
import math
import os
import re
import torch
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import folder_paths
import comfy.samplers

import comfy.samplers
import comfy.sample
import random
import time
import logging
from comfy.utils import ProgressBar
from comfy_extras.nodes_custom_sampler import Noise_RandomNoise, BasicScheduler, BasicGuider, SamplerCustomAdvanced
from comfy_extras.nodes_latent import LatentBatch
from comfy_extras.nodes_model_advanced import ModelSamplingFlux, ModelSamplingAuraFlow


# -------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------

def parse_string_to_list(input_string):
    try:
        if not input_string:
            return []
        items = input_string.replace('\n', ',').split(',')
        result = []
        for item in items:
            item = item.strip()
            if not item:
                continue
            try:
                num = float(item)
                if num.is_integer():
                    num = int(num)
                result.append(num)
            except ValueError:
                continue
        return result
    except:
        return []


def conditioning_set_values(conditioning, values):
    c = []
    for t in conditioning:
        n = [t[0], t[1].copy()]
        for k, v in values.items():
            if k == "guidance":
                n[1]['guidance_scale'] = v
        c.append(tuple(n))
    return c


def search_and_replace(text, extra_pnginfo, prompt):
    if extra_pnginfo is None or prompt is None:
        return text
    # if %date: in text, then replace with date
    #print(text)
    if '%date:' in text:
        for match in re.finditer(r'%date:(.*?)%', text):
            date_match = match.group(1)
            cursor = 0
            date_pattern = ''
            now = datetime.datetime.now()

            pattern_map = {
                'yyyy': now.strftime('%Y'),
                'yy': now.strftime('%y'),
                'MM': now.strftime('%m'),
                'M': now.strftime('%m').lstrip('0'),
                'dd': now.strftime('%d'),
                'd': now.strftime('%d').lstrip('0'),
                'hh': now.strftime('%H'),
                'h': now.strftime('%H').lstrip('0'),
                'mm': now.strftime('%M'),
                'm': now.strftime('%M').lstrip('0'),
                'ss': now.strftime('%S'),
                's': now.strftime('%S').lstrip('0')
            }

            sorted_keys = sorted(pattern_map.keys(), key=len, reverse=True)

            while cursor < len(date_match):
                replaced = False
                for key in sorted_keys:
                    if date_match.startswith(key, cursor):
                        date_pattern += pattern_map[key]
                        cursor += len(key)
                        replaced = True
                        break
                if not replaced:
                    date_pattern += date_match[cursor]
                    cursor += 1

            text = text.replace('%date:' + match.group(1) + '%', date_pattern)
    # Parse JSON if they are strings
    if isinstance(extra_pnginfo, str):
        extra_pnginfo = json.loads(extra_pnginfo)
    if isinstance(prompt, str):
        prompt = json.loads(prompt)

    # Map from "Node name for S&R" to id in the workflow
    node_to_id_map = {}
    try:
        for node in extra_pnginfo['workflow']['nodes']:
            node_name = node['properties'].get('Node name for S&R')
            node_id = node['id']
            node_to_id_map[node_name] = node_id
    except:
        return text

    # Find all patterns in the text that need to be replaced
    patterns = re.findall(r"%([^%]+)%", text)
    for pattern in patterns:
        # Split the pattern to get the node name and widget name
        node_name, widget_name = pattern.split('.')

        # Find the id for this node name
        node_id = node_to_id_map.get(node_name)
        if node_id is None:
            print(f"No node with name {node_name} found.")
            # check if user entered id instead of node name
            if node_name in node_to_id_map.values():
                node_id = node_name
            else:
                continue

        # Find the value of the specified widget in prompt JSON
        prompt_node = prompt.get(str(node_id))
        if prompt_node is None:
            print(f"No prompt data for node with id {node_id}.")
            continue

        widget_value = prompt_node['inputs'].get(widget_name)
        if widget_value is None:
            print(f"No widget with name {widget_name} found for node {node_name}.")
            continue

        # Replace the pattern in the text
        text = text.replace(f"%{pattern}%", str(widget_value))

    return text


# -------------------------------------------------------------
# Node: FluxTextSampler
# -------------------------------------------------------------

class FluxTextSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": ("MODEL", ),
            "conditioning": ("CONDITIONING", ),
            "latent_image": ("LATENT", ),
            "seed": ("INT", { "default": 0, "min": 0, "max": 0xffffffffffffffff }),
            "sampler": (comfy.samplers.KSampler.SAMPLERS, {
                "default": "euler",
                "multiselect": True
            }),
            "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {
                "default": "simple",
                "multiselect": True
            }),
            "steps": ("STRING", { "multiline": False, "dynamicPrompts": False, "default": "20" }),
            "guidance": ("STRING", { "multiline": False, "dynamicPrompts": False, "default": "3.5" }),
            "max_shift": ("STRING", { "multiline": False, "dynamicPrompts": False, "default": "" }),
            "base_shift": ("STRING", { "multiline": False, "dynamicPrompts": False, "default": "" }),
            "denoise": ("STRING", { "multiline": False, "dynamicPrompts": False, "default": "1.0" }),
        }}

    RETURN_TYPES = ("LATENT","SAMPLER_PARAMS","STRING")
    RETURN_NAMES = ("latent", "params","model_name")
    FUNCTION = "execute"
    CATEGORY = "🐈‍⬛ BK Utils/Sampling"

    def execute(self, model, conditioning, latent_image, seed, sampler, scheduler, steps, guidance, max_shift, base_shift, denoise):
        is_schnell = model.model.model_type == comfy.model_base.ModelType.FLOW
        model_name = getattr(model.model, 'name', None) or getattr(model, 'name', None)
        if model_name is None and hasattr(model.model, 'model_config'):
            model_name = model.model.model_config.get("model_name", "UnknownModel")

        noise = [seed]

        # --- Handle Sampler Input ---
        if isinstance(sampler, list):
            sampler = [s for s in sampler if s in comfy.samplers.KSampler.SAMPLERS]
        else:
            if sampler == '*':
                sampler = comfy.samplers.KSampler.SAMPLERS
            elif sampler.startswith("!"):
                sampler = sampler.replace("\n", ",").split(",")
                sampler = [s.strip("! ") for s in sampler]
                sampler = [s for s in comfy.samplers.KSampler.SAMPLERS if s not in sampler]
            else:
                sampler = sampler.replace("\n", ",").split(",")
                sampler = [s.strip() for s in sampler if s.strip() in comfy.samplers.KSampler.SAMPLERS]
        if not sampler:
            sampler = ['euler']

        # --- Handle Scheduler Input ---
        if isinstance(scheduler, list):
            scheduler = [s for s in scheduler if s in comfy.samplers.KSampler.SCHEDULERS]
        else:
            if scheduler == '*':
                scheduler = comfy.samplers.KSampler.SCHEDULERS
            elif scheduler.startswith("!"):
                scheduler = scheduler.replace("\n", ",").split(",")
                scheduler = [s.strip("! ") for s in scheduler]
                scheduler = [s for s in comfy.samplers.KSampler.SCHEDULERS if s not in scheduler]
            else:
                scheduler = scheduler.replace("\n", ",").split(",")
                scheduler = [s.strip() for s in scheduler if s in comfy.samplers.KSampler.SCHEDULERS]
        if not scheduler:
            scheduler = ['simple']

        # --- Parse numeric fields ---
        if steps == "":
            steps = "4" if is_schnell else "20"
        steps = parse_string_to_list(steps)
        denoise = "1.0" if denoise == "" else denoise
        denoise = parse_string_to_list(denoise)
        guidance = "3.5" if guidance == "" else guidance
        guidance = parse_string_to_list(guidance)

        if not is_schnell:
            max_shift = "1.15" if max_shift == "" else max_shift
            base_shift = "0.5" if base_shift == "" else base_shift
        else:
            max_shift = "0"
            base_shift = "1.0" if base_shift == "" else base_shift

        max_shift = parse_string_to_list(max_shift)
        base_shift = parse_string_to_list(base_shift)

        cond_encoded = [conditioning]

        out_latent = None
        out_params = []

        basicschedueler = BasicScheduler()
        basicguider = BasicGuider()
        samplercustomadvanced = SamplerCustomAdvanced()
        latentbatch = LatentBatch()
        modelsamplingflux = ModelSamplingFlux() if not is_schnell else ModelSamplingAuraFlow()

        width = latent_image["samples"].shape[3] * 8
        height = latent_image["samples"].shape[2] * 8

        total_samples = len(cond_encoded) * len(noise) * len(max_shift) * len(base_shift) * len(guidance) * len(sampler) * len(scheduler) * len(steps) * len(denoise)
        current_sample = 0
        if total_samples > 1:
            pbar = ProgressBar(total_samples)

        for conditioning in cond_encoded:
            for n in noise:
                randnoise = Noise_RandomNoise(n)
                for ms in max_shift:
                    for bs in base_shift:
                        if is_schnell:
                            work_model = modelsamplingflux.patch_aura(model, bs)[0]
                        else:
                            work_model = modelsamplingflux.patch(model, ms, bs, width, height)[0]
                        for g in guidance:
                            cond = conditioning_set_values(conditioning, {"guidance": g})
                            guider = basicguider.get_guider(work_model, cond)[0]
                            for s in sampler:
                                samplerobj = comfy.samplers.sampler_object(s)
                                for sc in scheduler:
                                    for st in steps:
                                        for d in denoise:
                                            sigmas = basicschedueler.get_sigmas(work_model, sc, st, d)[0]
                                            current_sample += 1
                                            logging.info(f"Sampling {current_sample}/{total_samples} with seed {n}, sampler {s}, scheduler {sc}, steps {st}, guidance {g}, max_shift {ms}, base_shift {bs}, denoise {d}")
                                            start_time = time.time()
                                            latent = samplercustomadvanced.sample(randnoise, guider, samplerobj, sigmas, latent_image)[1]
                                            elapsed_time = time.time() - start_time
                                            out_params.append({
                                                "time": elapsed_time,
                                                "seed": n,
                                                "width": width,
                                                "height": height,
                                                "sampler": s,
                                                "scheduler": sc,
                                                "steps": st,
                                                "guidance": g,
                                                "max_shift": ms,
                                                "base_shift": bs,
                                                "denoise": d,
                                            })

                                            if out_latent is None:
                                                out_latent = latent
                                            else:
                                                out_latent = latentbatch.batch(out_latent, latent)[0]
                                            if total_samples > 1:
                                                pbar.update(1)

        return (out_latent, out_params, model_name)


# -------------------------------------------------------------
# Node: FluxPromptSaver
# -------------------------------------------------------------

class FluxPromptSaver:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"

    @classmethod
    def INPUT_TYPES(s):
        return {
        "required": {
            "images": ("IMAGE",),
            "params": ("SAMPLER_PARAMS",),
            "positive": ("STRING", {"forceInput": True}),
            "model_name": ("STRING", {"forceInput": True}),
            "filename_prefix": ("STRING", {"default": "%date:yyyy-MM-dd%"}),
            "filename": ("STRING", {"default": "FLUX_%date:HHmmss%"}),
        },
        "optional": {
            "negative": ("STRING", {"forceInput": True}),
        }}

    RETURN_TYPES = ()
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "🐈‍⬛ BK Utils/Image"

    def save_images(self, images, params, positive, model_name, filename_prefix, filename, negative=""):
        filename_prefix = self.replace_dates(filename_prefix)
        filename = self.replace_dates(filename)

        results = []
        p = params[0]

        full_output = os.path.join(self.output_dir, filename_prefix)
        os.makedirs(full_output, exist_ok=True)

        for image in images:
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

            metadata = PngInfo()
            metadata.add_text("parameters",
                              self.create_metadata(p, positive, negative, model_name))

            file_base = filename
            file_ext = ".png"
            file_name = f"{file_base}{file_ext}"
            file_path = os.path.join(full_output, file_name)

            counter = 1
            while os.path.exists(file_path):
                file_name = f"{file_base}_{counter}{file_ext}"
                file_path = os.path.join(full_output, file_name)
                counter += 1

            img.save(file_path, pnginfo=metadata, optimize=True)
            results.append({
                "filename": file_name,
                "subfolder": filename_prefix,
                "type": self.type
            })

        return {"ui": {"images": results}}

    def replace_dates(self, s):
        pattern = re.compile(r'%date:(.*?)%')

        def repl(m):
            fmt = m.group(1)
            tokens = {
                'yyyy': '%Y',
                'MM': '%m',
                'dd': '%d',
                'HH': '%H',
                'mm': '%M',
                'ss': '%S',
            }
            for t, r in tokens.items():
                fmt = fmt.replace(t, r)
            try:
                return datetime.now().strftime(fmt)
            except:
                return m.group(0)

        return pattern.sub(repl, s)

    def create_metadata(self, params, positive, negative, model_name):
        sampler_scheduler = f"{params['sampler']}_{params['scheduler']}" if params['scheduler'] != 'normal' else params['sampler']
        negative_text = "(not used)" if not negative else negative
        guidance_val = params.get('guidance', 1.0)
        seed_val = params.get('seed', '?')

        return (
            f"{positive}\nNegative prompt: {negative_text}\n"
            f"Steps: {params['steps']}, Sampler: {sampler_scheduler}, CFG scale: {guidance_val}, "
            f"Seed: {seed_val}, Size: {params['width']}x{params['height']}, "
            f"Model: {model_name}, Version: ComfyUI"
        )


# -------------------------------------------------------------
# Node: ModelName
# -------------------------------------------------------------

class ModelName:
    @classmethod
    def INPUT_TYPES(s):
        model_list = []
        for folder in ["checkpoints", "models", "unet", "diffusion_models"]:
            try:
                model_list.extend(folder_paths.get_filename_list(folder))
            except:
                pass
        model_list = list(set(model_list))
        return {"required": {"model_name": (model_list,)}}

    RETURN_TYPES = ("STRING",)
    FUNCTION = "get_name"
    CATEGORY = "🐈‍⬛ BK Utils/Utils"

    def get_name(self, model_name):
        return (model_name,)



# -------------------------------------------------------------
# Node: SamePixelResolutionCalculator
# -------------------------------------------------------------

class SamePixelResolutionCalculator:
    """
    Calculates width and height with same pixel count as base square.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_size": ("INT", {"default": 1328, "min": 1, "max": 8192, "step": 8}),
                "aspect_preset": (["1:1","4:3","3:2","16:9","21:9","Custom"],),
                "custom_aspect_width": ("FLOAT", {"default": 16.0, "min": 1.0, "max": 100.0, "step": 0.1}),
                "custom_aspect_height": ("FLOAT", {"default": 9.0, "min": 1.0, "max": 100.0, "step": 0.1}),
                "round_multiple": (["none","8","64"],),
                "portrait_mode": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("INT","INT")
    RETURN_NAMES = ("width","height")
    FUNCTION = "calculate"
    CATEGORY = "🐈‍⬛ BK Utils/Resolution Tools"

    def calculate(self, base_size, aspect_preset, custom_aspect_width, custom_aspect_height, round_multiple, portrait_mode):
        presets = {
            "1:1": (1,1),
            "4:3": (4,3),
            "3:2": (3,2),
            "16:9": (16,9),
            "21:9": (21,9)
        }

        if aspect_preset in presets:
            a,b = presets[aspect_preset]
        else:
            a,b = custom_aspect_width, custom_aspect_height

        N = base_size * base_size
        W = round(math.sqrt(N * a / b))
        H = round(math.sqrt(N * b / a))

        if portrait_mode:
            W,H = H,W

        if round_multiple != "none":
            mult = int(round_multiple)
            W = round(W / mult) * mult
            H = round(H / mult) * mult

        return (W,H)


# -------------------------------------------------------------
# Node: IsOneOfGroupsActive
# -------------------------------------------------------------

class IsOneOfGroupsActive:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "group_name_contains": ("STRING", {"default": ""}),
                "active_state": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("boolean",)
    FUNCTION = "pass_state"
    CATEGORY = "🐈‍⬛ BK Utils/Logic"

    @classmethod
    def IS_CHANGED(cls, *args):
        return float("nan")

    def pass_state(self, group_name_contains, active_state):
        return (active_state,)


# -------------------------------------------------------------
# Node: DynamicGroupSwitchMulti
# -------------------------------------------------------------

class DynamicGroupSwitchMulti:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt_1": ("STRING", {"default": ""}),
                "group_1": ("STRING", {"default": ""}),
                "prompt_2": ("STRING", {"default": ""}),
                "group_2": ("STRING", {"default": ""}),
                "prompt_3": ("STRING", {"default": ""}),
                "group_3": ("STRING", {"default": ""}),
                "default": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output",)
    FUNCTION = "switch_input"
    CATEGORY = "🐈‍⬛ BK Utils/Logic"

    def switch_input(self, prompt_1, group_1, prompt_2, group_2, prompt_3, group_3, default):
        workflow = getattr(self, "graph", None)
        if workflow is None:
            return (default,)

        # Pair prompts with groups
        prompt_group_pairs = [
            (prompt_1, group_1),
            (prompt_2, group_2),
            (prompt_3, group_3),
        ]

        # Check each group in order
        for prompt, group_name in prompt_group_pairs:
            if not group_name:
                continue
            for node in workflow.nodes:
                if group_name.lower() in (node.group or "").lower():
                    if not getattr(node, "bypass", False):
                        return (prompt,)

        # Fallback
        return (default,)


# -------------------------------------------------------------
# Node: FileNameDefinition
# -------------------------------------------------------------

class FileNameDefinition:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {'date': (['true','false'], {'default':'true'}),
                             'date_directory': (['true','false'], {'default':'true'}),
                             'custom_directory': ('STRING', {'default': ''}),
                             'custom_text': ('STRING', {'default': ''})},
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},}

    RETURN_TYPES = ('STRING',)
    RETURN_NAMES = ('filename_prefix',)
    FUNCTION = 'get_filename_prefix'
    CATEGORY = '🐈‍⬛ BK Utils/Meta'

    def get_filename_prefix(self, date, date_directory, custom_directory, custom_text,
                            prompt=None, extra_pnginfo=None):
        filename_prefix = ''
        if date_directory == 'true':
            ts_str = datetime.datetime.now().strftime("%Y%m%d")
            filename_prefix += ts_str + '/'
        if custom_directory:
            custom_directory = search_and_replace(custom_directory, extra_pnginfo, prompt)
            filename_prefix += custom_directory + '/'
        if date == 'true':
            ts_str = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            filename_prefix += ts_str
        if custom_text != '':
            custom_text = search_and_replace(custom_text, extra_pnginfo, prompt)
            # remove invalid characters from filename
            custom_text = re.sub(r'[<>:"/\\|?*]', '', custom_text)
            filename_prefix += '_' + custom_text
        return (filename_prefix,)


# -------------------------------------------------------------
# Node: ZImageTurboResolutions
# -------------------------------------------------------------

class ZImageTurboResolutions:
    """
    A custom node for selecting turbo-optimized image resolutions
    """
    
    # Resolution mappings with all ratio options
    RESOLUTIONS = {
        "1024": [
            "1024x1024 (1:1)",
            "1152x896 (9:7)",
            "896x1152 (7:9)",
            "1152x864 (4:3)",
            "864x1152 (3:4)",
            "1248x832 (3:2)",
            "832x1248 (2:3)",
            "1280x720 (16:9)",
            "720x1280 (9:16)",
            "1344x576 (21:9)",
            "576x1344 (9:21)"
        ],
        "1280": [
            "1280x1280 (1:1)",
            "1440x1120 (9:7)",
            "1120x1440 (7:9)",
            "1472x1104 (4:3)",
            "1104x1472 (3:4)",
            "1536x1024 (3:2)",
            "1024x1536 (2:3)",
            "1536x864 (16:9)",
            "864x1536 (9:16)",
            "1680x720 (21:9)",
            "720x1680 (9:21)"
        ],
        "1536": [
            "1536x1536 (1:1)",
            "1728x1344 (9:7)",
            "1344x1728 (7:9)",
            "1728x1296 (4:3)",
            "1296x1728 (3:4)",
            "1872x1248 (3:2)",
            "1248x1872 (2:3)",
            "2048x1152 (16:9)",
            "1152x2048 (9:16)",
            "2016x864 (21:9)",
            "864x2016 (9:21)"
        ]
    }
    
    @classmethod
    def INPUT_TYPES(cls):
        # Get all resolution options
        resolution_options = list(cls.RESOLUTIONS.keys())
        
        # Get all ratio options (we'll show all, but filter in JS if needed)
        all_ratios = []
        for ratios in cls.RESOLUTIONS.values():
            all_ratios.extend(ratios)
        # Remove duplicates while preserving order
        all_ratios = list(dict.fromkeys(all_ratios))
        
        return {
            "required": {
                "resolution": (resolution_options, {"default": "1024"}),
                "ratio": (all_ratios, {"default": "1024x1024 (1:1)"}),
            },
        }
    
    RETURN_TYPES = ("INT", "INT", "INT")
    RETURN_NAMES = ("resolution", "width", "height")
    FUNCTION = "get_dimensions"
    CATEGORY = "🐈‍⬛ BK Utils/Resolution Tools"
    
    def get_dimensions(self, resolution, ratio):
        """
        Parse the selected resolution and ratio to return dimensions
        """
        # Extract width and height from the ratio string (format: "1024x1024 (1:1)")
        dimensions = ratio.split(" ")[0]  # Get "1024x1024" part
        width, height = dimensions.split("x")
        
        # Convert to integers
        resolution_int = int(resolution)
        width_int = int(width)
        height_int = int(height)
        
        return (resolution_int, width_int, height_int)


# Node registration
NODE_CLASS_MAPPINGS = {
    "ZImageTurboResolutions": ZImageTurboResolutions
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZImageTurboResolutions": "Z-Image Turbo Resolutions"
}



# -------------------------------------------------------------
#  Node Registration (ALL nodes merged here)
# -------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "FluxPromptSaver": FluxPromptSaver,
    "FluxTextSampler": FluxTextSampler,
    "ModelName": ModelName,
    "IsOneOfGroupsActive": IsOneOfGroupsActive,
    "DynamicGroupSwitchMulti": DynamicGroupSwitchMulti,
    "SamePixelResolutionCalculator": SamePixelResolutionCalculator,
    "FileNameDefinition": FileNameDefinition,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxPromptSaver": "🐈‍⬛ Flux Prompt Saver",
    "FluxTextSampler": "🐈‍⬛ Flux Text Sampler",
    "ModelName": "🐈‍⬛ Model Name",
    "IsOneOfGroupsActive": "🐈‍⬛ IsOneOfGroupsActive",
    "DynamicGroupSwitchMulti": "🐈‍⬛ Dynamic Group Switch (3-way)",
    "SamePixelResolutionCalculator": "🐈‍⬛ Same Pixel Resolution Calculator",
    "FileNameDefinition": "🐈‍⬛ File Name Definition",
    "ZImageTurboResolutions": "🐈‍⬛ Z-Image Turbo Resolutions",
}

