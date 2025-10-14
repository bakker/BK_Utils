# BK_Utils - a utility collection for ComfyUI
## Nodes
Currently only one node, the "Collection" part is WIP


### Same Pixel Resolution Calculator
In order to generate images with the same amount of pixels, but varying aspect ratio (starting from the premise that the amount of pixels defines generation time).
#### Inputs
1. Base dimension in pixels
2. Predefined aspect ratio:
    1. 1:1
    2. 4:3
    3. 3:2
    4. 16:9
    5. 21:9
    6. Custom
3. Custom Width factor (integer)
4. Custom Height factor (integer)
5. round_multiple: to round the result to nearest 64, 8 or no rounding
6. Portrait Mode: self-explanatory 
#### Outputs
1. Width
2. Height
Connect these to an "Empty Latent" node.
<img width="314" height="235" alt="Screenshot_586" src="https://github.com/user-attachments/assets/4d559e1f-c7a7-4910-bae4-168e86f6aa00" />
