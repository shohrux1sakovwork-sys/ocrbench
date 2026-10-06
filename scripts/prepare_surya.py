"""Make a transformers-v4-loadable copy of datalab-to/surya-ocr-2 in models/.

The published tokenizer_config.json uses the transformers v5 format
(tokenizer_class "TokenizersBackend", extra_special_tokens as a list), which
the transformers 4.x used by the local vLLM cannot load. This links every file
of the downloaded snapshot and writes a v4 tokenizer_config.json next to them;
the tokenizer itself (tokenizer.json) is unchanged.

    python scripts/prepare_surya.py    # -> models/surya-ocr-2
"""

from __future__ import annotations

import json
from pathlib import Path

from huggingface_hub import snapshot_download

REPO = "datalab-to/surya-ocr-2"
OUT = Path(__file__).resolve().parent.parent / "models" / "surya-ocr-2"


def main() -> None:
    src = Path(snapshot_download(REPO, local_files_only=True))
    OUT.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        dst = OUT / f.name
        if f.name == "tokenizer_config.json" or dst.exists() or dst.is_symlink():
            continue
        dst.symlink_to(f.resolve())

    cfg = json.loads((src / "tokenizer_config.json").read_text(encoding="utf-8"))
    cfg["tokenizer_class"] = "Qwen2TokenizerFast"
    extra = cfg.pop("extra_special_tokens", None)
    if isinstance(extra, list):
        cfg["additional_special_tokens"] = extra
    for key in ("backend", "is_local"):
        cfg.pop(key, None)
    (OUT / "tokenizer_config.json").write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(OUT)


if __name__ == "__main__":
    main()
