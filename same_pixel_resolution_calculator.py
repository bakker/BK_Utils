import math

class SamePixelResolutionCalculator:
    """
    Calculates width and height for a given aspect ratio,
    preserving total pixel count of a base square image.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_size": ("INT", {
                    "default": 1328,
                    "min": 1,
                    "max": 8192,
                    "step": 1
                }),
                "aspect_preset": ([
                    "1:1",
                    "4:3",
                    "3:2",
                    "16:9",
                    "21:9",
                    "Custom"
                ],),
                "custom_aspect_width": ("FLOAT", {
                    "default": 16.0,
                    "min": 1.0,
                    "max": 100.0,
                    "step": 0.1
                }),
                "custom_aspect_height": ("FLOAT", {
                    "default": 9.0,
                    "min": 1.0,
                    "max": 100.0,
                    "step": 0.1
                }),
                "round_multiple": (["none", "8", "64"],),
                "portrait_mode": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("INT", "INT",)
    RETURN_NAMES = ("width", "height",)
    FUNCTION = "calculate"
    CATEGORY = "BK Utils/Resolution Tools"

    def calculate(
        self,
        base_size,
        aspect_preset,
        custom_aspect_width,
        custom_aspect_height,
        round_multiple,
        portrait_mode
    ):
        # Determine aspect ratio
        presets = {
            "1:1": (1, 1),
            "4:3": (4, 3),
            "3:2": (3, 2),
            "16:9": (16, 9),
            "21:9": (21, 9)
        }

        if aspect_preset in presets:
            a, b = presets[aspect_preset]
        else:
            a, b = custom_aspect_width, custom_aspect_height

        # Base pixel count (square)
        N = base_size * base_size

        # Calculate W and H preserving total pixel count
        W = math.sqrt(N * a / b)
        H = math.sqrt(N * b / a)

        # Round to integers
        W, H = round(W), round(H)

        # Swap for portrait orientation if needed
        if portrait_mode:
            W, H = H, W

        # Optionally round to nearest multiple of 8 or 64
        if round_multiple != "none":
            mult = int(round_multiple)
            W = round(W / mult) * mult
            H = round(H / mult) * mult

        return (W, H)


# Register node for ComfyUI
NODE_CLASS_MAPPINGS = {
    "SamePixelResolutionCalculator": SamePixelResolutionCalculator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SamePixelResolutionCalculator": "Same Pixel Resolution Calculator"
}
