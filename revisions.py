"""Hugging Face revisions (commit hashes) of the checkpoints used for the paper.

The loaders in binding_patch*.py and grader.py pass these as `revision=`, so a default run fetches exactly the
paper's weights. A model id that is not listed here loads the repository's current revision.
"""

REVISION = {
    "sd-legacy/stable-diffusion-v1-5": "451f4fe16113bff5a5d2269ed5ad43b0592e9a14",
    "PixArt-alpha/PixArt-Sigma-XL-2-512-MS": "76fb7eb5a9314bc1e4e479d2f13447517fca9be4",
    "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS": "e102b3591cc82e97071b8b4cb90d834d0c487207",
    "stabilityai/stable-diffusion-3.5-medium": "b940f670f0eda2d07fbb75229e779da1ad11eb80",
    "Qwen/Qwen2.5-VL-7B-Instruct": "cc594898137f460bfe9f0759e9844b3ce807cfb5",
    "llava-hf/llava-1.5-7b-hf": "b234b804b114d9e37bb655e11cbbb5f5e971b7a9",
    "openai/clip-vit-large-patch14": "32bd64288804d66eefd0ccbe215aa642df71cc41",  # sanity-check grader only
}
