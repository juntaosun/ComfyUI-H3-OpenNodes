import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import re


class CNTextOverlay:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "text": ("STRING", {"multiline": False, "default": "video-1"}),
                "x_margin": ("INT", {"default": 50, "min": 0, "max": 8192, "step": 1}),
                "y_margin": ("INT", {"default": 20, "min": 0, "max": 8192, "step": 1}),
                "font_size": ("INT", {"default": 50, "min": 8, "max": 256, "step": 1}),
                "font_color": ("COLOR", {"default": "#abdb29"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "add_text"
    CATEGORY = "PainterNodes"

    def _parse_color(self, color_str):
        color_str = str(color_str).strip().lower()
        color_map = {
            "white": (255, 255, 255),
            "black": (0, 0, 0),
            "red": (255, 0, 0),
            "green": (0, 255, 0),
            "blue": (0, 0, 255),
            "yellow": (255, 255, 0),
            "cyan": (0, 255, 255),
            "magenta": (255, 0, 255),
            "gray": (128, 128, 128),
            "grey": (128, 128, 128),
        }
        if color_str in color_map:
            return color_map[color_str]
        if color_str.startswith("#"):
            hex_color = color_str.lstrip("#")
            if len(hex_color) == 6:
                return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            elif len(hex_color) == 3:
                return tuple(int(hex_color[i]*2, 16) for i in range(3))
        m = re.match(r"(\d+),\s*(\d+),\s*(\d+)", color_str)
        if m:
            r, g, b = int(m.group(1)), int(m.group(2)), int(m.group(3))
            return (max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)))
        return (255, 255, 255)

    def _get_font(self, size=32):
        font_paths = [
            "C:/Windows/Fonts/msyh.ttc",
            "C:/Windows/Fonts/msyh.ttf",
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/segoeui.ttf",
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/STHeiti Light.ttc",
            "/System/Library/Fonts/Helvetica.ttc",
            "/Library/Fonts/Arial.ttf",
        ]
        for path in font_paths:
            if os.path.exists(path):
                try:
                    return ImageFont.truetype(path, size)
                except Exception:
                    continue
        return ImageFont.load_default()

    def add_text(self, image, text, x_margin, y_margin, font_size, font_color):
        if not text or text.strip() == "":
            return (image,)

        font = self._get_font(font_size)
        color = self._parse_color(font_color)
        shadow_color = (0, 0, 0)
        if color == (0, 0, 0):
            shadow_color = (255, 255, 255)

        results = []
        if image.dim() == 3:
            image = image.unsqueeze(0)

        batch_size = image.shape[0]

        for i in range(batch_size):
            img_tensor = image[i]
            np_img = (img_tensor.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)

            if np_img.shape[-1] == 3:
                pil_img = Image.fromarray(np_img, mode="RGB")
            else:
                pil_img = Image.fromarray(np_img)

            draw = ImageDraw.Draw(pil_img)

            draw.text((x_margin + 1, y_margin + 1), text, fill=shadow_color, font=font)
            draw.text((x_margin, y_margin), text, fill=color, font=font)

            result_np = np.array(pil_img).astype(np.float32) / 255.0
            results.append(torch.from_numpy(result_np))

        output = torch.stack(results, dim=0).to(device=image.device, dtype=image.dtype)
        return (output,)


NODE_CLASS_MAPPINGS = {
    "CNTextOverlay": CNTextOverlay,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CNTextOverlay": "CN Text Overlay",
}
NODE_REGISTRY = {
    "classes": NODE_CLASS_MAPPINGS,
    "names": NODE_DISPLAY_NAME_MAPPINGS,
}