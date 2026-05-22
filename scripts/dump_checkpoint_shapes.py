import argparse
from pathlib import Path

import torch


def get_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict", "module", "net"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    return checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="tmp/swin_tiny_patch4_window7_224.pth",
        type=str,
    )
    parser.add_argument(
        "--output",
        default="tmp/swin_tiny_patch4_window7_224_shapes.txt",
        type=str,
    )
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state_dict = get_state_dict(checkpoint)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for name, value in state_dict.items():
            if torch.is_tensor(value):
                shape = "x".join(str(dim) for dim in value.shape)
            else:
                shape = type(value).__name__
            f.write(f"{name}\t{shape}\n")


if __name__ == "__main__":
    main()
