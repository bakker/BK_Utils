# 🐈‍⬛ BK_Utils

A collection of utility nodes for ComfyUI: Flux sampling sweeps, prompt-aware saving, resolution helpers, file-name building, batch logging and a text library for system prompts and wildcard lists.

All nodes show up in the node menu under **🐈‍⬛ BK Utils**.

## Contents

| Node | Category | What it does |
|---|---|---|
| [Flux Text Sampler](#flux-text-sampler) | Sampling | Samples every combination of samplers, schedulers, steps, guidance, shift and denoise in one go |
| [Flux Prompt Saver](#flux-prompt-saver) | Image | Saves PNGs with A1111-style `parameters` metadata |
| [Model Name](#model-name) | Utils | Outputs a model file name as a string |
| [Same Pixel Resolution Calculator](#same-pixel-resolution-calculator) | Resolution Tools | Width/height for any aspect ratio at a constant pixel count |
| [Z-Image-Turbo Resolutions](#z-image-turbo-resolutions) | Resolution Tools | Preset resolutions for Z-Image-Turbo at three size tiers |
| [File Name Definition](#file-name-definition) | Meta | Builds a `filename_prefix` from dates, folders and values from other nodes |
| [Batch Log Writer](#batch-log-writer) | Meta | Appends one CSV row per run, for evaluating large batches |
| [Text Library Selector](#text-library-selector) | LLM | Loads system prompts and wildcard lists from files, with seeded `{a\|b\|c}` expansion |
| [IsOneOfGroupsActive](#isoneofgroupsactive) | Logic | Boolean pass-through for group-based switching |
| [Dynamic Group Switch (3-way)](#dynamic-group-switch-3-way) | Logic | Picks one of three strings depending on which group is active |

## Installation

Clone the repository into ComfyUI's `custom_nodes` folder and restart ComfyUI:

```
cd ComfyUI/custom_nodes
git clone https://github.com/bakker/BK_Utils
```

There are no extra requirements: everything the nodes use (torch, numpy, Pillow) already ships with ComfyUI. On startup the console lists every node that was loaded:

```
[🐈‍⬛ BK_Utils] Loading 10 node(s) from BK_Utils.py:
  - FluxPromptSaver (🐈‍⬛ Flux Prompt Saver)
  ...
```

---

## Sampling

### Flux Text Sampler

Runs a full parameter sweep for Flux in a single node. Every field that takes text accepts a comma- or newline-separated list, and the node samples **every combination** of the values you give it. All results come out as one latent batch.

#### Inputs

| Input | Type | Default | Notes |
|---|---|---|---|
| `model` | MODEL | | |
| `conditioning` | CONDITIONING | | |
| `latent_image` | LATENT | | Width and height are read from this latent |
| `seed` | INT | 0 | One seed per run |
| `sampler` | multiselect | `euler` | Pick one or more samplers |
| `scheduler` | multiselect | `simple` | Pick one or more schedulers |
| `steps` | list | `20` | e.g. `20, 25, 30`. Empty = 20 (4 for FLOW-type models such as Schnell) |
| `guidance` | list | `3.5` | Flux guidance, e.g. `2.5, 3.5` |
| `max_shift` | list | empty | Empty = 1.15. Ignored for FLOW-type models |
| `base_shift` | list | empty | Empty = 0.5 (1.0 for FLOW-type models, where it is used as the AuraFlow shift) |
| `denoise` | list | `1.0` | |

When `sampler` or `scheduler` arrives as text instead of a selection, it also understands:

- `*` — every available sampler/scheduler
- `!name1, name2` — everything **except** the names listed
- `name1, name2` — only those (unknown names are skipped)

If nothing valid remains, the node falls back to `euler` / `simple`.

#### Outputs

| Output | Type | Notes |
|---|---|---|
| `latent` | LATENT | All samples, batched |
| `params` | SAMPLER_PARAMS | One entry per sample: time taken, seed, width, height, sampler, scheduler, steps, guidance, max_shift, base_shift, denoise |
| `model_name` | STRING | Best-effort model name |

#### Tips

- The number of samples is the product of all list lengths: 3 samplers × 2 schedulers × 3 step counts = 18 images. A progress bar appears when there is more than one.
- Each sample is logged to the console with its parameters, so you can match images to settings.
- For non-FLOW models, `ModelSamplingFlux` is applied with the latent's width and height; for FLOW-type models, `ModelSamplingAuraFlow` is used instead.

---

## Image

### Flux Prompt Saver

Saves images as PNG with a `parameters` text chunk in the A1111 format, so sites and tools that read A1111 metadata (Civitai, image browsers) can show prompt, seed, sampler and model.

#### Inputs

| Input | Type | Default | Notes |
|---|---|---|---|
| `images` | IMAGE | | |
| `params` | SAMPLER_PARAMS | | From Flux Text Sampler |
| `positive` | STRING (input) | | Positive prompt text |
| `model_name` | STRING (input) | | e.g. from Model Name or Flux Text Sampler |
| `filename_prefix` | STRING | `%date:yyyy-MM-dd%` | Used as the subfolder inside the output directory |
| `filename` | STRING | `FLUX_%date:HHmmss%` | File name without extension |
| `negative` | STRING (optional) | | Written as `(not used)` when empty |

Date tokens in `filename_prefix` and `filename`: `yyyy`, `MM`, `dd`, `HH`, `mm`, `ss`.

Existing files are never overwritten: a counter is added (`FLUX_142530_1.png`, `_2`, …).

The metadata written looks like:

```
<positive prompt>
Negative prompt: (not used)
Steps: 20, Sampler: euler_simple, CFG scale: 3.5, Seed: 123, Size: 1024x1024, Model: flux1-dev, Version: ComfyUI
```

The scheduler is appended to the sampler name unless it is `normal`. Flux guidance is written as `CFG scale`.

#### Notes

- The metadata of the **first** `params` entry is used for every image in the batch. With a multi-combination sweep from Flux Text Sampler, the per-image settings are therefore not all correct; use it with single-combination runs, or check the console log.
- Only the A1111 `parameters` chunk is written; the ComfyUI workflow is not embedded.

---

## Utils

### Model Name

Pick a model from a dropdown and get its name as a STRING, for metadata or file names. The list combines the `checkpoints`, `unet` and `diffusion_models` folders.

| Output | Type | Notes |
|---|---|---|
| `STRING` | STRING | The file name as listed, including subfolder and extension |

---

## Resolution Tools

### Same Pixel Resolution Calculator

Generates width and height with the same number of pixels as a square of `base_size`, at any aspect ratio. Since pixel count largely decides generation time and VRAM use, this lets you change aspect ratio without changing cost.

#### Inputs

| Input | Type | Default | Notes |
|---|---|---|---|
| `base_size` | INT | 1328 | Side of the reference square (pixel count = base_size²) |
| `aspect_preset` | choice | | `1:1`, `4:3`, `3:2`, `16:9`, `21:9`, `Custom` |
| `custom_aspect_width` | FLOAT | 16.0 | Used with `Custom` only |
| `custom_aspect_height` | FLOAT | 9.0 | Used with `Custom` only |
| `round_multiple` | choice | | `none`, `8` or `64` |
| `portrait_mode` | BOOLEAN | false | Swaps width and height |

#### Outputs

`width` and `height` (INT). Connect them to an Empty Latent Image node.

Example: `base_size` 1328 at 16:9 with rounding to 64 gives 1792 × 1024.

<img width="314" height="235" alt="Same Pixel Resolution Calculator" src="https://github.com/user-attachments/assets/4d559e1f-c7a7-4910-bae4-168e86f6aa00" />

### Z-Image-Turbo Resolutions

Preset resolutions for Z-Image-Turbo in three size tiers. Picking a tier filters the `ratio` dropdown to the resolutions of that tier.

| Ratio | 1024 | 1280 | 1536 |
|---|---|---|---|
| 1:1 | 1024×1024 | 1280×1280 | 1536×1536 |
| 9:7 / 7:9 | 1152×896 | 1440×1120 | 1728×1344 |
| 4:3 / 3:4 | 1152×864 | 1472×1104 | 1728×1296 |
| 3:2 / 2:3 | 1248×832 | 1536×1024 | 1872×1248 |
| 16:9 / 9:16 | 1280×720 | 1536×864 | 2048×1152 |
| 21:9 / 9:21 | 1344×576 | 1680×720 | 2016×864 |

Every landscape ratio also has its portrait counterpart in the dropdown.

#### Outputs

| Output | Type | Notes |
|---|---|---|
| `resolution` | INT | The selected tier (1024, 1280 or 1536) |
| `width` | INT | |
| `height` | INT | |

---

## Meta

### File Name Definition

Builds a `filename_prefix` for any saver node. The parts are joined in this order:

```
[YYYYMMDD/][custom_directory/][YYYYMMDDHHMMSS][_custom_text]
```

#### Inputs

| Input | Default | Notes |
|---|---|---|
| `date` | true | Adds a `YYYYMMDDHHMMSS` timestamp |
| `date_directory` | true | Adds a `YYYYMMDD/` folder |
| `custom_directory` | empty | Extra folder; placeholders allowed |
| `custom_text` | empty | Appended after `_`; placeholders allowed; characters invalid in file names (`<>:"/\|?*`) are removed |

#### Placeholders

`custom_directory` and `custom_text` support:

- **`%date:<format>%`** — tokens `yyyy`, `yy`, `MM` (month), `M`, `dd`, `d`, `hh` (24-hour), `h`, `mm` (minutes), `m`, `ss`, `s`. Single-letter tokens drop the leading zero. Example: `%date:yyyy-MM-dd%`.
- **`%Node.widget%`** — the value of a widget on another node. `Node` is matched, in this order, against:
  1. the node's *Node name for S&R* property
  2. a node id, including subgraph ids such as `18:1`
  3. the node title
  4. the node class type

  Model file names are cleaned to the bare name, so `%UNETLoader.unet_name%` with `MOSS\flux1-dev.safetensors` gives `flux1-dev`. Values coming from a link instead of a widget can't be read; a message is printed to the console and the placeholder is left as is.

Example: `custom_text` = `%KSampler.seed%_%Load Checkpoint.ckpt_name%` → `20261116/20261116141500_123456_sdxl_base`.

### Batch Log Writer

Appends one row per run to a CSV file, so a large batch (say 50–60 images) can be evaluated afterwards in Excel. Each row holds:

```
timestamp; seed; wildcards; llm_prompt; verdict
```

`verdict` is left empty for you to fill in while reviewing.

#### Inputs

| Input | Type | Default | Notes |
|---|---|---|---|
| `wildcards` | STRING (input) | | e.g. the wildcard picks for this run |
| `llm_prompt` | STRING (input) | | The prompt the LLM produced |
| `seed` | INT (input) | | |
| `filename_prefix` | STRING | `logs/batch` | Wire the same prefix your saver gets; only its **folder** part is used |
| `log_name` | STRING | `batch_log` | File name without `.csv` |
| `delimiter` | choice | `;` | `;` is what Excel expects with Belgian/Dutch regional settings; `,` for everything else |

#### Notes

- The node runs every time, even with identical inputs.
- A new file gets a header row and a UTF-8 BOM, so Excel shows accents correctly.
- Paths are resolved the same way ComfyUI's own image saver does, so a symlinked output directory works. A path that would end up outside the output directory raises an error.

---

## LLM

### Text Library Selector

Loads a system prompt or a wildcard list from a text file, so only the **file name** is stored in the workflow, never the text itself. That keeps shared workflows clean and lets you edit prompts in your own editor.

#### Where the files go

```
<models>/text_library/
├── system_prompts/     → category "System Prompts"
│   └── gladiator.txt
└── wildcard_lists/     → category "Wildcard Lists"
    └── LUDI_GLADIATORII.txt
```

- `.txt` and `.md` files are listed, including those in subfolders.
- Both folders are created automatically on first start.
- Because `text_library` is registered as a model folder, a symlinked `models` directory works, and you can add more locations in `extra_model_paths.yaml`:

  ```yaml
  my_library:
    base_path: D:/AI/
    text_library: text_library
  ```

#### Inputs

| Input | Notes |
|---|---|
| `category` | `System Prompts` or `Wildcard Lists`; the `file` list is filtered to that category |
| `file` | The file to load. New files show up when you switch category, without restarting ComfyUI |
| `seed` | Seed for wildcard expansion; has *control after generate* |

#### Outputs

| Output | Notes |
|---|---|
| `text` | The file contents as is |
| `expanded` | The same text with every `{a\|b\|c}` group replaced by one option, chosen by `seed` |

Expansion handles nested groups (`{red {dark\|light}\|blue}`) and always gives the same result for the same seed. Braces without a `|` are left alone, and `__wildcard__` file references are not expanded here (leave those to a wildcard node such as Impact Pack's).

#### Notes

- The node re-runs when the selected file is edited or the seed changes.
- A file saved with a BOM (common with Windows editors) is read correctly.
- When a category has no files, the list shows `(no files found)` and running the workflow gives an error naming the folder to look in.

---

## Logic

### IsOneOfGroupsActive

Outputs a BOOLEAN for driving switches, with a `group_name_contains` field to note which group(s) it refers to.

| Input | Default | Notes |
|---|---|---|
| `group_name_contains` | empty | Group name, or part of it |
| `active_state` | false | Passed straight through to the output |

> **Status:** the node currently passes `active_state` through unchanged; it does not read group state itself yet. Set `active_state` by hand or from another node. It re-runs on every queue.

### Dynamic Group Switch (3-way)

Meant to output one of three strings depending on which group is active: it checks `group_1`, `group_2` and `group_3` in order and returns the matching `prompt_n` for the first group (matched by partial, case-insensitive name) that has a node that isn't bypassed. If none match, `default` is returned.

| Input | Notes |
|---|---|
| `prompt_1` … `prompt_3` | Text to output for each group |
| `group_1` … `group_3` | Group name, or part of it; empty = skip |
| `default` | Fallback text |

> **Status:** experimental. The node needs access to the workflow graph, which ComfyUI does not currently give a Python node, so in practice it returns `default`.

---

## Front-end extensions

Two small JavaScript extensions improve the dropdowns:

- `text_library_selector.js` — filters the **Text Library Selector** file list by category and refreshes it from the server, so new files appear without a reload.
- `z_image_turbo_resolutions.js` — filters the **Z-Image-Turbo Resolutions** ratio list by the selected tier.

ComfyUI loads them from the `js` folder (`WEB_DIRECTORY = "./js"` in `__init__.py`).
