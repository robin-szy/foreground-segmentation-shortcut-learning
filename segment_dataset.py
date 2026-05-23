from __future__ import annotations

import argparse
import csv
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from typing import Any

import clip


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create segmented iNaturalist images with SAM + CLIP")

    # Experiment details
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default="inaturalist_12K")
    parser.add_argument("--out-dir", type=str, default="inaturalist_12K_segmented")
    parser.add_argument("--splits", nargs="+", choices=["train", "val", "test"], default=["train"]) #default=["train", "val"])
    parser.add_argument("--seed", type=int, default=420)

    # Debugging / output control
    parser.add_argument("--max-images-per-class", type=int, default=None)   # None is default
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug-side-by-side", action="store_true")
    parser.add_argument("--save-masks", action="store_true", default=True)
    parser.add_argument("--no-save-masks", dest="save_masks", action="store_false")

    # Mask handling
    parser.add_argument("--score-threshold", type=float, default=0.0)   # Below this threshold, take whole image
    parser.add_argument("--background", choices=["white", "black", "transparent"], default="white")
    parser.add_argument("--crop-to-bbox", action="store_true")  # Crop final output to the selected mask bbox. Default keeps original image size
    parser.add_argument("--min-area-ratio", type=float, default=0.003)
    parser.add_argument("--max-area-ratio", type=float, default=0.90)    # "Ignore masks larger than this fraction of the image area."
    parser.add_argument("--expand-mask", action="store_true", default=True)     # Erode the mask
    parser.add_argument("--no-expand-mask", dest="expand_mask", action="store_false")
    parser.add_argument("--expand-max", type=int, default=3)    # Some settings for eroding the mask
    parser.add_argument("--expand-p", type=float, default=0.5)
    parser.add_argument("--expand-iterations", type=int, default=10)

    # Prompt handling
    parser.add_argument("--prompt-mode", choices=["global", "class_specific", "class_plus_global"], default="class_plus_global")
    parser.add_argument("--extra-good-prompts", nargs="*", default=[])
    parser.add_argument("--extra-bad-prompts", nargs="*", default=[])

    # Models
    parser.add_argument("--sam-model-type", choices=["vit_b", "vit_l", "vit_h"], default="vit_b")
    parser.add_argument("--sam-checkpoint", type=str, default="sam_vit_b_01ec64.pth")
    parser.add_argument("--clip-model", type=str, default="ViT-B/32")
    parser.add_argument("--points-per-side", type=int, default=8)  # HPC 32, home 8
    parser.add_argument("--pred-iou-thresh", type=float, default=0.90)
    parser.add_argument("--stability-score-thresh", type=float, default=0.80)
    parser.add_argument("--min-mask-region-area", type=int, default=0)

    # Runtime
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true", help="Use autocast for CLIP image encoding on CUDA.")

    args = parser.parse_args()

    return args


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

CLASSES = [
    "Amphibia",
    "Animalia",
    "Arachnida",
    "Aves",
    "Fungi",
    "Insecta",
    "Mammalia",
    "Mollusca",
    "Plantae",
    "Reptilia",
]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASSES)}

GOOD_PROMPTS_GLOBAL = [
    "a photo of an animal",
    "a photo of a living organism",
    "a photo of wildlife",
    "a close-up photo of an organism",
    "a photo of an animal body",
    "a photo of an animal head",
    "a photo of an animal on a white background",
    "a photo of a centipede",
    "a photo of a millipede",
    "a photo of an amphibian",
    "a photo of a frog",
    "a photo of a toad",
    "a photo of a salamander",
    "a photo of an arachnid",
    "a photo of a spider",
    "a photo of a scorpion",
    "a photo of a bird",
    "a photo of feathers",
    "a photo of a bird head",
    "a photo of a fungus",
    "a photo of a mushroom",
    "a photo of fungi",
    "a photo of an insect",
    "a photo of a beetle",
    "a photo of a butterfly",
    "a photo of a bee",
    "a photo of a fly",
    "a photo of a mammal",
    "a photo of a furry animal",
    "a photo of an animal face",
    "a photo of a mollusk",
    "a photo of a snail",
    "a photo of a slug",
    "a photo of a shellfish",
    "a photo of a plant",
    "a photo of a flower",
    "a photo of leaves",
    "a photo of a stem",
    "a photo of a reptile",
    "a photo of a lizard",
    "a photo of a snake",
    "a photo of a turtle",
]

GOOD_PROMPTS_BY_CLASS: Dict[str, List[str]] = {
    "Amphibia": [
        "a photo of an amphibian",
        "a photo of a frog",
        "a photo of a toad",
        "a photo of a salamander",
        "a close-up photo of an animal",
    ],
    "Animalia": [
        "a photo of an animal",
        "a photo of wildlife",
        "a photo of an animal body",
        "a photo of an animal head",
        "a photo of a centipede",
        "a photo of a millipede",
    ],
    "Arachnida": [
        "a photo of an arachnid",
        "a photo of a spider",
        "a photo of a scorpion",
        "a close-up photo of an animal",
    ],
    "Aves": [
        "a photo of a bird",
        "a photo of feathers",
        "a photo of a bird head",
        "a photo of an animal",
    ],
    "Fungi": [
        "a photo of a fungus",
        "a photo of a mushroom",
        "a photo of fungi",
        "a photo of a living organism",
    ],
    "Insecta": [
        "a photo of an insect",
        "a photo of a beetle",
        "a photo of a butterfly",
        "a photo of a bee",
        "a photo of a fly",
    ],
    "Mammalia": [
        "a photo of a mammal",
        "a photo of a furry animal",
        "a photo of an animal face",
        "a photo of an animal",
    ],
    "Mollusca": [
        "a photo of a mollusk",
        "a photo of a snail",
        "a photo of a slug",
        "a photo of a shellfish",
    ],
    "Plantae": [
        "a photo of a plant",
        "a photo of a flower",
        "a photo of leaves",
        "a photo of a stem",
    ],
    "Reptilia": [
        "a photo of a reptile",
        "a photo of a lizard",
        "a photo of a snake",
        "a photo of a turtle",
    ],
}

BAD_PROMPTS_GLOBAL = [
    "a photo of a person",
    "a photo of a human hand",
    "a photo of fingers",
    "a photo of a face",
    "a photo of clothing",
    "a photo of a camera strap",
    "a photo of a cage",
    "a photo of a fence",
    "a photo of a wall",
    "a photo of a building",
    "a photo of a road",
    "a photo of a path",
    "a photo of a table",
    "a photo of a label",
    "a photo of a sign",
    "a photo of text",
    "a photo of a watermark",
    "a photo of a landscape",
    "a photo of scenery",
    "a blurry background",
    "an out of focus background",
    "an empty background",
    "a plain background",
    "a photo of grass",
    "a photo of a forest",
    "a photo of bushes",
    "a photo of tree bark",
    "a photo of a tree branch",
    "a photo of wood",
    "a photo of dead leaves",
    "a photo of soil",
    "a photo of mud",
    "a photo of rocks",
    "a photo of sand",
    "a photo of moss",
    "a photo of a lake",
    "a photo of the ocean",
    "a photo of a river",
    "a photo of water",
    "a photo of water reflections",
    "a photo of the water surface",
    "a photo of the sky",
    "a photo of blue sky",
    "a photo of clouds",
    "a photo of a shadow",
    "a photo of a reflection",
    "a dark silhouette",
    "a bright light",
    "a blurry object",
    "a white background",
    "a black background",
    "a photo of the background",
    "a cropped background region",
    "a close-up texture",
    "a patch of grass",
    "a patch of water",
    "a piece of wood",
]


@dataclass
class SegmentationResult:
    source_path: str
    output_path: str
    mask_path: str
    split: str
    label: str
    status: str
    best_score: float
    best_prompt: str
    best_prompt_prob: float
    best_mask_idx: int
    num_masks: int
    mask_area_ratio: float
    bbox: List[int]
    seconds: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def list_images(class_dir: Path) -> List[Path]:
    if not class_dir.exists():
        return []
    return sorted(p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def relative_to_data(path: Path, data_dir: Path) -> str:
    return str(path.relative_to(data_dir)).replace("\\", "/")


def setup_sam(args: argparse.Namespace, device: torch.device) -> Any:
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

    checkpoint = Path(args.sam_checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"SAM checkpoint not found: {checkpoint}. Download the checkpoint and pass --sam-checkpoint."
        )

    sam = sam_model_registry[args.sam_model_type](checkpoint=str(checkpoint))
    sam.to(device)
    sam.eval()

    return SamAutomaticMaskGenerator(
        sam,
        points_per_side=args.points_per_side,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
        min_mask_region_area=args.min_mask_region_area,
    )


def random_expand_mask(mask: np.ndarray, max_expand: int = 3, p: float = 0.3, iterations: int = 1) -> np.ndarray:
    mask_u8 = mask.astype(np.uint8)
    h, w = mask_u8.shape
    kernel = np.ones((3, 3), np.uint8)

    for _ in range(iterations):
        eroded = cv2.erode(mask_u8, kernel, iterations=1)
        boundary = mask_u8 - eroded
        new_mask = mask_u8.copy()
        ys, xs = np.where(boundary > 0)

        for y, x in zip(ys, xs):
            if np.random.rand() > p:
                continue
            dy = np.random.randint(-1, 2)
            dx = np.random.randint(-1, 2)
            if dy == 0 and dx == 0:
                continue
            length = np.random.randint(1, max_expand + 1)
            for step in range(1, length + 1):
                ny = y + dy * step
                nx = x + dx * step
                if 0 <= ny < h and 0 <= nx < w:
                    new_mask[ny, nx] = 1
        mask_u8 = new_mask

    return mask_u8.astype(bool)


def get_prompts(label: str, args: argparse.Namespace) -> Tuple[List[str], List[str]]:
    if args.prompt_mode == "global":
        good = list(GOOD_PROMPTS_GLOBAL)
    elif args.prompt_mode == "class_specific":
        good = list(GOOD_PROMPTS_BY_CLASS[label])
    elif args.prompt_mode == "class_plus_global":
        # Class prompts first, then broad prompts. dict.fromkeys preserves order and removes duplicates
        good = list(dict.fromkeys(GOOD_PROMPTS_BY_CLASS[label] + GOOD_PROMPTS_GLOBAL))
    else:
        raise ValueError(args.prompt_mode)

    good.extend(args.extra_good_prompts)
    bad = list(BAD_PROMPTS_GLOBAL) + list(args.extra_bad_prompts)
    return good, bad


def encode_text_prompts(
    clip_model,
    prompts: Sequence[str],
    device: torch.device,
) -> torch.Tensor:

    text = clip.tokenize(list(prompts)).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(text)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features


def apply_mask(image_rgb: np.ndarray, mask: np.ndarray, background: str) -> Image.Image:
    if background == "transparent":
        rgba = np.dstack([image_rgb, (mask.astype(np.uint8) * 255)])
        return Image.fromarray(rgba, mode="RGBA")

    fill_value = 255 if background == "white" else 0
    out = np.full_like(image_rgb, fill_value)
    out[mask] = image_rgb[mask]
    return Image.fromarray(out, mode="RGB")


def save_image(img: Image.Image, path: Path, background: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if background == "transparent":
        # JPEG cannot store alpha, so use PNG for transparent outputs.
        path = path.with_suffix(".png")
    img.save(path)


def make_side_by_side(original: Image.Image, segmented: Image.Image, path: Path) -> None:
    orig_rgb = original.convert("RGB")
    seg_rgb = segmented.convert("RGB")
    w = orig_rgb.width + seg_rgb.width
    h = max(orig_rgb.height, seg_rgb.height)
    canvas = Image.new("RGB", (w, h), color=(255, 255, 255))
    canvas.paste(orig_rgb, (0, 0))
    canvas.paste(seg_rgb, (orig_rgb.width, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def score_masks(
    image_rgb: np.ndarray,
    masks: List[dict],
    good_prompts: List[str],
    bad_prompts: List[str],
    text_features: torch.Tensor,
    clip_model,
    preprocess,
    device: torch.device,
    args: argparse.Namespace,
) -> Tuple[int, float, str, float, np.ndarray, List[int]]:

    prompts = good_prompts + bad_prompts
    n_good = len(good_prompts)
    h, w = image_rgb.shape[:2]
    image_area = h * w

    best_idx = -1
    best_score = -float("inf")
    best_prompt = ""
    best_prompt_prob = 0.0
    best_mask: Optional[np.ndarray] = None
    best_bbox = [0, 0, w, h]

    for idx, mask_data in enumerate(masks):
        raw_mask = mask_data["segmentation"].astype(bool)
        area_ratio = float(raw_mask.sum()) / float(image_area)
        if area_ratio < args.min_area_ratio or area_ratio > args.max_area_ratio:
            continue

        score_mask = raw_mask
        if args.expand_mask:
            score_mask = random_expand_mask(
                score_mask,
                max_expand=args.expand_max,
                p=args.expand_p,
                iterations=args.expand_iterations,
            )

        masked = apply_mask(image_rgb, score_mask, background="white").convert("RGB")
        x, y, bw, bh = [int(v) for v in mask_data["bbox"]]
        crop = masked.crop((x, y, x + bw, y + bh))
        if crop.width == 0 or crop.height == 0:
            continue

        img_input = preprocess(crop).unsqueeze(0).to(device)
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                image_features = clip_model.encode_image(img_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            similarity = image_features @ text_features.T
            probs = (100.0 * similarity).softmax(dim=-1)[0]

        score = float(probs[:n_good].sum().item() - probs[n_good:].sum().item())
        top_idx = int(probs.argmax().item())
        top_prompt = prompts[top_idx]
        top_prob = float(probs[top_idx].item())

        if score > best_score:
            best_score = score
            best_idx = idx
            best_prompt = top_prompt
            best_prompt_prob = top_prob
            best_mask = raw_mask
            best_bbox = [x, y, bw, bh]

    if best_mask is None:
        return -1, -float("inf"), "", 0.0, np.ones((h, w), dtype=bool), [0, 0, w, h]

    if best_prompt in {"a photo of a spider", "a photo of an arachnid"}:
        best_mask = ~best_mask

    return best_idx, best_score, best_prompt, best_prompt_prob, best_mask, best_bbox


def process_one_image(
    img_path: Path,
    label: str,
    split: str,
    data_dir: Path,
    out_dir: Path,
    mask_generator: Any,
    clip_model,
    preprocess,
    text_cache: Dict[str, Tuple[List[str], List[str], torch.Tensor]],
    device: torch.device,
    args: argparse.Namespace,
) -> SegmentationResult:

    start = time.time()
    rel = relative_to_data(img_path, data_dir)
    out_path = out_dir / rel
    if args.background == "transparent":
        out_path = out_path.with_suffix(".png")
    mask_path = out_dir / "masks" / Path(rel).with_suffix(".png")
    debug_path = out_dir / "debug_side_by_side" / rel

    expected_path = mask_path if args.save_masks else debug_path

    if expected_path.exists() and not args.overwrite:
        return SegmentationResult(
            source_path=str(img_path), output_path=str(out_path), mask_path=str(mask_path), split=split, label=label,
            status="skipped_exists", best_score=float("nan"), best_prompt="", best_prompt_prob=float("nan"),
            best_mask_idx=-1, num_masks=-1, mask_area_ratio=float("nan"), bbox=[], seconds=time.time() - start,
        )

    pil_original = Image.open(img_path).convert("RGB")
    image_rgb = np.array(pil_original)

    masks = mask_generator.generate(image_rgb)

    if label not in text_cache:
        good, bad = get_prompts(label, args)
        prompts = good + bad
        text_cache[label] = (good, bad, encode_text_prompts(clip_model, prompts, device))
    good_prompts, bad_prompts, text_features = text_cache[label]

    best_idx, best_score, best_prompt, best_prompt_prob, best_mask, bbox = score_masks(
        image_rgb=image_rgb,
        masks=masks,
        good_prompts=good_prompts,
        bad_prompts=bad_prompts,
        text_features=text_features,
        clip_model=clip_model,
        preprocess=preprocess,
        device=device,
        args=args,
    )

    status = "segmented"
    if best_idx < 0 or best_score < args.score_threshold:
        status = "fallback_original"
        best_mask = np.ones(image_rgb.shape[:2], dtype=bool)
        segmented = pil_original.copy()
        bbox = [0, 0, pil_original.width, pil_original.height]
    else:
        segmented = apply_mask(image_rgb, best_mask, args.background)
        if args.crop_to_bbox:
            x, y, bw, bh = bbox
            segmented = segmented.crop((x, y, x + bw, y + bh))

    #out_path.parent.mkdir(parents=True, exist_ok=True)
    #segmented.save(out_path)
    if args.save_masks:
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray((best_mask.astype(np.uint8) * 255), mode="L").save(mask_path)
    if args.debug_side_by_side:
        make_side_by_side(pil_original, segmented, debug_path)

    mask_area_ratio = float(best_mask.sum()) / float(best_mask.size)
    return SegmentationResult(
        source_path=str(img_path),
        output_path=str(out_path),
        mask_path=str(mask_path),
        split=split,
        label=label,
        status=status,
        best_score=float(best_score),
        best_prompt=best_prompt,
        best_prompt_prob=float(best_prompt_prob),
        best_mask_idx=int(best_idx),
        num_masks=len(masks),
        mask_area_ratio=mask_area_ratio,
        bbox=[int(v) for v in bbox],
        seconds=time.time() - start,
    )


def write_metadata_row(path: Path, row: SegmentationResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    data = asdict(row)
    # Store bbox as JSON string to keep CSV simple.
    data["bbox"] = json.dumps(data["bbox"])
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(data.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(data)


def collect_images(args: argparse.Namespace) -> List[Tuple[Path, str, str]]:
    data_dir = Path(args.data_dir)
    items: List[Tuple[Path, str, str]] = []
    rng = random.Random(args.seed)

    for split in args.splits:
        for label in CLASSES:
            class_dir = data_dir / split / label
            images = list_images(class_dir)
            if args.max_images_per_class is not None:
                rng.shuffle(images)
                images = sorted(images[: args.max_images_per_class])
            for img_path in images:
                items.append((img_path.resolve(), label, split))

    return items




def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    data_dir = Path(args.data_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    if args.run_name:
        out_dir = out_dir / args.run_name

    if not data_dir.exists():
        raise FileNotFoundError(f"data-dir not found: {data_dir}")

    device = torch.device(args.device)
    print("\n================ SEGMENTATION CONFIG ================")
    for key, value in vars(args).items():
        print(f"{key:24}: {value}")
    print(f"resolved_data_dir       : {data_dir}")
    print(f"resolved_out_dir        : {out_dir}")
    print(f"device                  : {device}")
    if device.type == "cuda":
        print(f"gpu                     : {torch.cuda.get_device_name(0)}")
    print("=====================================================\n")

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "segment_args.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)
    items = collect_images(args)
    print(f"Found {len(items)} images to process.")
    if not items:
        return

    mask_generator = setup_sam(args, device)

    clip_model, preprocess = clip.load(args.clip_model, device=device)
    clip_model.eval()

    text_cache: Dict[str, Tuple[List[str], List[str], torch.Tensor]] = {}
    metadata_path = out_dir / "segmentation_metadata.csv"

    counts: Dict[str, int] = {}
    t0 = time.time()
    for i, (img_path, label, split) in enumerate(items, start=1):
        try:
            result = process_one_image(
                img_path=img_path,
                label=label,
                split=split,
                data_dir=data_dir,
                out_dir=out_dir,
                mask_generator=mask_generator,
                clip_model=clip_model,
                preprocess=preprocess,
                text_cache=text_cache,
                device=device,
                args=args,
            )
            counts[result.status] = counts.get(result.status, 0) + 1
            write_metadata_row(metadata_path, result)
            print(
                f"[{i:05d}/{len(items):05d}] {split}/{label}/{img_path.name} | "
                f"{result.status} | score={result.best_score:.4f} | "
                f"prompt={result.best_prompt!r} | masks={result.num_masks} | {result.seconds:.1f}s",
                flush=True,
            )
        except Exception as exc:
            counts["error"] = counts.get("error", 0) + 1
            print(f"[{i:05d}/{len(items):05d}] ERROR {img_path}: {exc}", flush=True)
            error_row = SegmentationResult(
                source_path=str(img_path), output_path="", mask_path="", split=split, label=label,
                status=f"error: {type(exc).__name__}: {exc}", best_score=float("nan"), best_prompt="",
                best_prompt_prob=float("nan"), best_mask_idx=-1, num_masks=-1, mask_area_ratio=float("nan"),
                bbox=[], seconds=0.0,
            )
            write_metadata_row(metadata_path, error_row)

    summary = {
        "total_images": len(items),
        "counts": counts,
        "seconds_total": time.time() - t0,
        "out_dir": str(out_dir),
    }
    with (out_dir / "segmentation_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("\nSummary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
