from enum import Enum
from typing_extensions import override
import torch
from comfy_api.latest import ComfyExtension, io


class AspectRatio(str, Enum):
    SQUARE = "1:1 (Square)"
    PHOTO_V = "2:3 (Portrait Photo)"
    PHOTO_H = "3:2 (Photo)"
    STANDARD_V = "3:4 (Portrait Standard)"
    STANDARD_H = "4:3 (Standard)"
    WIDESCREEN_V = "9:16 (Portrait Widescreen)"
    WIDESCREEN_H = "16:9 (Widescreen)"
    ULTRAWIDE_H = "21:9 (Ultrawide)"


ASPECT_RATIOS: dict[AspectRatio, tuple[int, int]] = {
    AspectRatio.SQUARE: (1, 1),
    AspectRatio.PHOTO_V: (2, 3),
    AspectRatio.PHOTO_H: (3, 2),
    AspectRatio.STANDARD_V: (3, 4),
    AspectRatio.STANDARD_H: (4, 3),
    AspectRatio.WIDESCREEN_V: (9, 16),
    AspectRatio.WIDESCREEN_H: (16, 9),
    AspectRatio.ULTRAWIDE_H: (21, 9),
}

# multiple 下拉可选值，默认对齐到 8 像素。
# 标准视频使用 8
# wan 模型使用 16
# H3 模型使用 32
MULTIPLE_OPTIONS = ["8", "16", "32"]


def calculate_resolution(aspect_ratio: str, long_side: int, multiple: int) -> tuple[int, int]:
    """根据所选比例和长边，计算对齐到指定倍数后的宽度与高度。"""
    w_ratio, h_ratio = ASPECT_RATIOS[aspect_ratio]
    long_side = int(long_side)
    multiple = int(multiple)
    if long_side <= 0:
        raise ValueError("long_side 必须大于 0，当前为 %s" % (long_side,))
    if multiple <= 0:
        raise ValueError("multiple 必须大于 0，当前为 %s" % (multiple,))

    # 先把长边对齐到倍数，再按比例计算短边并同样对齐。
    long_aligned = max(multiple, round(long_side / multiple) * multiple)
    if w_ratio >= h_ratio:
        width = long_aligned
        height = max(multiple, round(long_aligned * h_ratio / w_ratio / multiple) * multiple)
    else:
        height = long_aligned
        width = max(multiple, round(long_aligned * w_ratio / h_ratio / multiple) * multiple)
    return int(width), int(height)


def make_black_preview(width: int, height: int) -> torch.Tensor:
    """创建用于预览尺寸的纯黑图像，形状为 [1, H, W, 3]。"""
    return torch.zeros((1, int(height), int(width), 3), dtype=torch.float32)


def make_empty_latent(width: int, height: int, batch_size: int = 1) -> dict:
    """创建与 EmptyLatentImage 一致的空 latent，形状为 [B, 4, H//8, W//8]。"""
    try:
        import comfy.model_management as model_management
        device = model_management.intermediate_device()
        dtype = model_management.intermediate_dtype()
    except Exception:
        device = None
        dtype = torch.float32

    latent = torch.zeros(
        [int(batch_size), 4, int(height) // 8, int(width) // 8],
        device=device,
        dtype=dtype,
    )
    return {"samples": latent, "downscale_ratio_spacial": 8}


class ImageResolution(io.ComfyNode):
    """根据宽高比和图像长边计算宽度、高度，并输出纯黑预览图与空 latent。"""

    @classmethod
    def define_schema(cls):
        """定义分辨率选择节点的输入输出，含长边、对齐倍数、预览图和空 latent。"""
        return io.Schema(
            node_id="ImageResolution",
            display_name="Image Resolution Selector",
            category="open/utilities",
            description="Calculate width and height from aspect ratio and long side. Also outputs a black preview image and empty latent.",
            inputs=[
                io.Combo.Input(
                    "aspect_ratio",
                    options=AspectRatio,
                    default=AspectRatio.SQUARE,
                    tooltip="The aspect ratio for the output dimensions.",
                ),
                io.Combo.Input(
                    "multiple",
                    options=MULTIPLE_OPTIONS,
                    default="8",
                    tooltip="宽高对齐到该倍数。",
                ),
                io.Int.Input(
                    "long_side",
                    default=1024,
                    min=8,
                    max=16384,
                    step=8,
                    tooltip="图像长边像素。短边按所选比例等比计算。",
                ),
            ],
            outputs=[
                io.Image.Output(
                    "empty_image", tooltip="纯黑预览图，尺寸与计算得到的宽高一致，仅用于连接预览。"
                ),
                io.Latent.Output(
                    "empty_latent", tooltip="与计算宽高对应的空 latent，空间下采样比为 8。"
                ),
                io.Int.Output(
                    "width", tooltip="Calculated width in pixels aligned to the selected multiple."
                ),
                io.Int.Output(
                    "height", tooltip="Calculated height in pixels aligned to the selected multiple."
                ),
            ],
        )

    @classmethod
    def execute(cls, 
                aspect_ratio: str, 
                multiple,
                long_side: int, 
                ) -> io.NodeOutput:
        """按长边和比例计算宽高，并返回纯黑预览图与空 latent。"""
        width, height = calculate_resolution(aspect_ratio, long_side, int(multiple))
        image = make_black_preview(width, height)
        latent = make_empty_latent(width, height)
        return io.NodeOutput(image, latent, width, height)


class ImageResolutionExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        """返回本扩展注册的节点列表。"""
        return [
            ImageResolution,
        ]


async def comfy_entrypoint() -> ImageResolutionExtension:
    """ComfyUI 加载本扩展时的入口。"""
    return ImageResolutionExtension()


NODE_CLASS_MAPPINGS = {
    "ImageResolution": ImageResolution,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageResolution": "Image Resolution (Custom)",
}
NODE_REGISTRY = {
    "classes": NODE_CLASS_MAPPINGS,
    "names": NODE_DISPLAY_NAME_MAPPINGS,
}
