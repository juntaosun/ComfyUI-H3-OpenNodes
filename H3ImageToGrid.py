"""H3 Image To Grid：将视频帧均分成网格后拼接，并按长边等比缩放。

典型用途：把一批数量不固定的视频帧，按 rows x cols 抽成代表帧，
再拼成一张网格预览图。最终输出图像的最长边等于 long_side，
短边按原始网格比例缩放。帧数不足格子数时，空格填充纯白。
"""

import torch

import comfy.utils


def _validate_grid(rows, cols, long_side):
    """校验网格行列和长边参数，返回整型 (rows, cols, long_side, cell_count)。"""
    rows = int(rows)
    cols = int(cols)
    long_side = int(long_side)
    if rows <= 0 or cols <= 0:
        raise ValueError("rows 和 cols 必须都大于 0，当前为 %sx%s" % (rows, cols))
    if long_side <= 0:
        raise ValueError("long_side 必须大于 0，当前为 %s" % (long_side,))
    return rows, cols, long_side, rows * cols


def _select_even_indices(total, cell_count):
    """将 total 帧平均分成格子并返回对应帧索引。

    帧数不少于格子数时，按 cell_count 份均分并取每份第一帧。
    帧数不足时不复用，只返回已有帧的索引，空格由调用方填白色。
    """
    total = int(total)
    cell_count = int(cell_count)
    if total <= 0:
        raise ValueError("images 至少需要 1 帧")
    if cell_count <= 0:
        raise ValueError("网格格子数必须大于 0")
    fill_count = min(total, cell_count)
    return [i * total // fill_count for i in range(fill_count)]


def _normalize_frame_size(images):
    """校验 IMAGE 批次形状，Comfy 批次帧尺寸已一致时原样返回。"""
    if images is None or getattr(images, "ndim", 0) != 4:
        raise ValueError("images 期望形状为 [B, H, W, C]")
    if int(images.shape[0]) <= 0:
        raise ValueError("images 至少需要 1 帧")
    if int(images.shape[1]) <= 0 or int(images.shape[2]) <= 0:
        raise ValueError("输入图像宽高必须大于 0")
    return images


def _make_white_frame(images):
    """按输入帧尺寸创建纯白色占位图。"""
    return torch.ones(
        (int(images.shape[1]), int(images.shape[2]), int(images.shape[3])),
        dtype=images.dtype,
        device=images.device,
    )


def _select_even_frames(images, cell_count):
    """从输入批次中按均分规则取出帧，不足格子数时用纯白图补齐。"""
    images = _normalize_frame_size(images)
    indices = _select_even_indices(int(images.shape[0]), cell_count)
    frames = [images[index] for index in indices]
    if len(frames) < cell_count:
        white = _make_white_frame(images)
        frames.extend(white.clone() for _ in range(cell_count - len(frames)))
    return torch.stack(frames, dim=0)


def _tile_images(frames, rows, cols):
    """按行优先（先左到右，再上到下）把帧拼成一张网格图。"""
    rows, cols, _, cell_count = _validate_grid(rows, cols, 1)
    if int(frames.shape[0]) != cell_count:
        raise ValueError(
            "拼接帧数必须等于 rows*cols=%s，当前为 %s"
            % (cell_count, int(frames.shape[0])))

    row_tensors = []
    for row in range(rows):
        start = row * cols
        row_frames = [frames[start + col] for col in range(cols)]
        row_tensors.append(torch.cat(row_frames, dim=1))
    return torch.cat(row_tensors, dim=0)


def _scale_to_long_side(image, long_side):
    """按长边等比缩放单张图像，短边由原始宽高比计算。"""
    long_side = int(long_side)
    if long_side <= 0:
        raise ValueError("long_side 必须大于 0")

    squeezed = False
    if image.ndim == 3:
        image = image.unsqueeze(0)
        squeezed = True
    if image.ndim != 4:
        raise ValueError("缩放输入期望 [H, W, C] 或 [B, H, W, C]")

    height = int(image.shape[1])
    width = int(image.shape[2])
    if height <= 0 or width <= 0:
        raise ValueError("拼接后的图像宽高必须大于 0")

    if width >= height:
        target_width = long_side
        target_height = max(1, int(round(long_side * height / width)))
    else:
        target_height = long_side
        target_width = max(1, int(round(long_side * width / height)))

    if target_width == width and target_height == height:
        return image.squeeze(0) if squeezed else image

    samples = image.movedim(-1, 1)
    samples = comfy.utils.common_upscale(
        samples, target_width, target_height, "lanczos", "disabled")
    scaled = samples.movedim(1, -1)
    return scaled.squeeze(0) if squeezed else scaled


def images_to_grid(images, cols=5, rows=2, long_side=1536):
    """将输入视频帧均分成网格并缩放到指定长边，返回 [1, H, W, C]。"""
    rows, cols, long_side, cell_count = _validate_grid(rows, cols, long_side)
    frames = _select_even_frames(images, cell_count)
    grid = _tile_images(frames, rows, cols)
    scaled = _scale_to_long_side(grid, long_side)
    if scaled.ndim == 3:
        scaled = scaled.unsqueeze(0)
    return scaled


class H3ImageToGrid:
    """把数量不固定的视频帧均分成网格图，并按长边等比输出。"""

    @classmethod
    def INPUT_TYPES(cls):
        """声明节点输入：图像批次、列数、行数和输出长边。"""
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "视频帧图像批次 [B,H,W,C]。"
                               "会按 rows*cols 均分成对应份数后各取一帧；"
                               "帧数不足时剩余格子填充纯白。"}),
                "cols": ("INT", {
                    "default": 5, "min": 1, "max": 64, "step": 1,
                    "tooltip": "网格列数，从左到右排列。默认 5。"}),
                "rows": ("INT", {
                    "default": 2, "min": 1, "max": 64, "step": 1,
                    "tooltip": "网格行数，从上到下排列。默认 2。"}),
                "long_side": ("INT", {
                    "default": 1536, "min": 1, "max": 8192, "step": 1,
                    "tooltip": "拼接完成后的最长边像素。短边按网格原比例缩放。"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "to_grid"
    CATEGORY = "H3/image"
    DESCRIPTION = (
        "Split a variable-length IMAGE batch into rows*cols even segments, "
        "take the first frame of each segment, tile them left-to-right and "
        "top-to-bottom, then scale the stitched image so its longest side "
        "equals long_side. Empty cells are filled with solid white when "
        "there are fewer frames than grid cells.")

    def to_grid(self, images, cols=5, rows=2, long_side=1536):
        """执行均分抽帧、网格拼接和长边缩放。"""
        return (images_to_grid(images, cols=cols, rows=rows, long_side=long_side),)


NODE_CLASS_MAPPINGS = {
    "H3ImageToGrid": H3ImageToGrid,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3ImageToGrid": "H3 Image To Grid",
}
NODE_REGISTRY = {
    "classes": NODE_CLASS_MAPPINGS,
    "names": NODE_DISPLAY_NAME_MAPPINGS,
}
