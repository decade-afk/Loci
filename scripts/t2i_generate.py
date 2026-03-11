#!/usr/bin/env python
"""
Minimal local text-to-image runner for Loci CLI.

Usage example:
  python scripts/t2i_generate.py \
    --prompt "a watercolor cat astronaut" \
    --model-id hf-internal-testing/tiny-stable-diffusion-pipe \
    --output outputs/t2i.png \
    --steps 4 \
    --guidance-scale 0.0
"""

import argparse
import os
import sys
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local text-to-image generation")
    parser.add_argument("--prompt", required=True, help="Text prompt")
    parser.add_argument("--model-id", required=True, help="HF model id or local path")
    parser.add_argument("--output", required=True, help="Output image path")
    parser.add_argument("--steps", type=int, default=4, help="Denoising steps")
    parser.add_argument("--guidance-scale", type=float, default=0.0, help="CFG scale")
    parser.add_argument("--width", type=int, default=None, help="Optional output width")
    parser.add_argument("--height", type=int, default=None, help="Optional output height")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed")
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Execution device",
    )
    parser.add_argument(
        "--fallback-placeholder",
        action="store_true",
        help="Generate a deterministic placeholder image when model load/inference fails",
    )
    return parser.parse_args()


def env_flag(name: str) -> bool:
    value = os.getenv(name, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def save_placeholder_image(
    output_path: str,
    prompt: str,
    model_id: str,
    width: int | None,
    height: int | None,
    reason: str,
) -> int:
    try:
        from PIL import Image, ImageDraw
    except Exception as exc:
        print(
            f"[t2i] fallback requested but Pillow is unavailable: {exc}",
            file=sys.stderr,
        )
        return 3

    w = width if width and width > 0 else 512
    h = height if height and height > 0 else 512
    image = Image.new("RGB", (w, h), (22, 36, 58))
    draw = ImageDraw.Draw(image)
    lines = [
        "LOCI T2I FALLBACK",
        f"reason: {reason}",
        f"model: {model_id}",
        f"prompt: {prompt[:96]}",
    ]
    y = 16
    for line in lines:
        draw.text((16, y), line, fill=(240, 240, 240))
        y += 22
    image.save(output_path)
    print("[t2i] WARNING: fallback placeholder image generated")
    print(f"[t2i] saved image: {output_path}")
    return 0


def main() -> int:
    args = parse_args()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fallback_enabled = args.fallback_placeholder or env_flag("LOCI_T2I_FALLBACK")

    try:
        import torch
        from diffusers import StableDiffusionPipeline
    except Exception as exc:  # pragma: no cover - runtime dependency guard
        if fallback_enabled:
            return save_placeholder_image(
                output_path=args.output,
                prompt=args.prompt,
                model_id=args.model_id,
                width=args.width,
                height=args.height,
                reason=f"dependency import failed: {exc}",
            )
        print(
            "Missing dependencies for text-to-image. "
            "Install with: python -m pip install -r scripts/requirements-t2i.txt",
            file=sys.stderr,
        )
        print(f"Import error: {exc}", file=sys.stderr)
        return 2

    use_cuda = args.device == "cuda" and torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"[t2i] loading model: {args.model_id}")
    print(f"[t2i] device={device} dtype={torch_dtype}")

    start = time.time()
    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            args.model_id,
            torch_dtype=torch_dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )
    except Exception as exc:
        if fallback_enabled:
            return save_placeholder_image(
                output_path=args.output,
                prompt=args.prompt,
                model_id=args.model_id,
                width=args.width,
                height=args.height,
                reason=f"model load failed: {exc}",
            )
        raise

    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    generator = None
    if args.seed is not None:
        generator = torch.Generator(device=device).manual_seed(args.seed)
        print(f"[t2i] seed={args.seed}")

    kwargs = {
        "prompt": args.prompt,
        "num_inference_steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "generator": generator,
    }
    if args.width is not None:
        kwargs["width"] = args.width
    if args.height is not None:
        kwargs["height"] = args.height

    try:
        image = pipe(**kwargs).images[0]
    except Exception as exc:
        if fallback_enabled:
            return save_placeholder_image(
                output_path=args.output,
                prompt=args.prompt,
                model_id=args.model_id,
                width=args.width,
                height=args.height,
                reason=f"inference failed: {exc}",
            )
        raise

    image.save(args.output)

    elapsed = time.time() - start
    print(f"[t2i] saved image: {args.output}")
    print(f"[t2i] elapsed: {elapsed:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
