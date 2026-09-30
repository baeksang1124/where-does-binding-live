"""Automated binding graders.

QwenGrader  -- PRIMARY grader: per-object color VQA with Qwen2.5-VL-7B ("what color is the
               cube?"), loaded in 4-bit NF4 (bitsandbytes); directly measures binding.
LlavaGrader -- SECOND grader: the same per-object questions with LLaVA-1.5-7B (a different
               model family), used for the dual-grader checks.
CLIPGrader  -- sanity check only: whole-image caption similarity, flip = sim(swapped) - sim(clean).
"""
import torch
from PIL import Image
from revisions import REVISION


class CLIPGrader:
    def __init__(self, model_id="openai/clip-vit-large-patch14", device="cuda"):
        from transformers import CLIPModel, CLIPProcessor
        self.device = device
        self.model = CLIPModel.from_pretrained(model_id, revision=REVISION.get(model_id)).to(device).eval()
        self.proc = CLIPProcessor.from_pretrained(model_id, revision=REVISION.get(model_id))

    @torch.no_grad()
    def _img_feat(self, img):
        inp = self.proc(images=img, return_tensors="pt").to(self.device)
        f = self.model.get_image_features(**inp)
        return torch.nn.functional.normalize(f, dim=-1)

    @torch.no_grad()
    def _txt_feat(self, texts):
        inp = self.proc(text=texts, return_tensors="pt", padding=True).to(self.device)
        f = self.model.get_text_features(**inp)
        return torch.nn.functional.normalize(f, dim=-1)

    @torch.no_grad()
    def flip_score(self, img, clean_caption, swapped_caption):
        """>0 => image looks more like the SWAPPED binding; <0 => more like clean."""
        imf = self._img_feat(img)
        tf = self._txt_feat([clean_caption, swapped_caption])
        sims = (imf @ tf.T).squeeze(0)  # [clean, swapped]
        return float(sims[1] - sims[0])


class LlavaGrader:
    """Independent 2nd grader: LLaVA-1.5-7B (different family from Qwen), same _ask interface."""
    def __init__(self, model_id="llava-hf/llava-1.5-7b-hf", device="cuda"):
        from transformers import LlavaForConditionalGeneration, AutoProcessor
        self.device = device
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_id, revision=REVISION.get(model_id), torch_dtype=torch.float16, device_map=device).eval()
        self.proc = AutoProcessor.from_pretrained(model_id, revision=REVISION.get(model_id))

    @torch.no_grad()
    def _ask(self, img, question):
        conv = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
        prompt = self.proc.apply_chat_template(conv, add_generation_prompt=True)
        inp = self.proc(images=img, text=prompt, return_tensors="pt").to(self.device, torch.float16)
        out = self.model.generate(**inp, max_new_tokens=8, do_sample=False)
        ans = self.proc.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)
        return ans.strip().lower()

    @torch.no_grad()
    def _ask_color(self, img, obj):
        return self._ask(img, f"What color is the {obj} in this image? "
                              "Answer with a single color word.")


class QwenGrader:
    """Per-object color VQA via Qwen2.5-VL-7B."""
    def __init__(self, model_id="Qwen/Qwen2.5-VL-7B-Instruct", device="cuda"):
        from transformers import (Qwen2_5_VLForConditionalGeneration, AutoProcessor,
                                  BitsAndBytesConfig)
        self.device = device
        qcfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                  bnb_4bit_compute_dtype=torch.bfloat16,
                                  bnb_4bit_use_double_quant=True)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, revision=REVISION.get(model_id), quantization_config=qcfg, device_map=device).eval()
        # cap vision tokens to keep memory/latency low (512px images don't need more)
        self.proc = AutoProcessor.from_pretrained(
            model_id, revision=REVISION.get(model_id), min_pixels=256 * 28 * 28, max_pixels=768 * 28 * 28)

    @torch.no_grad()
    def _ask(self, img, question):
        """Generic single-object attribute VQA. `question` should already name the object."""
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": question}]}]
        text = self.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = self.proc(text=[text], images=[img], return_tensors="pt").to(self.device)
        out = self.model.generate(**inp, max_new_tokens=8, do_sample=False)
        gen = out[:, inp.input_ids.shape[1]:]
        ans = self.proc.batch_decode(gen, skip_special_tokens=True)[0].strip().lower()
        return ans

    @torch.no_grad()
    def _ask_color(self, img, obj):
        return self._ask(img, f"What color is the {obj} in this image? "
                              "Answer with a single color word.")

    @torch.no_grad()
    def binding_swap_score(self, img, pair):
        """Fraction of the two objects whose color matches the SWAPPED binding.
        clean binding: o1=c1, o2=c2 ; swapped: o1=c2, o2=c1.  Returns in [0,1] (0/.5/1)."""
        a1 = self._ask_color(img, pair["o1"])
        a2 = self._ask_color(img, pair["o2"])
        s = 0.0
        # object o1 should be c1 (clean) or c2 (swapped)
        if pair["c2"] in a1 and pair["c1"] not in a1:
            s += 0.5
        if pair["c1"] in a2 and pair["c2"] not in a2:
            s += 0.5
        return s, (a1, a2)

    @torch.no_grad()
    def object_swap_score(self, img, obj, clean_color, swap_color):
        """Binding measure on ONE object: 1.0 if it reads the SWAPPED color, 0.0 if the
        CLEAN color, 0.5 if ambiguous/neither. Lower variance than the two-object score."""
        a = self._ask_color(img, obj)
        has_swap = swap_color in a
        has_clean = clean_color in a
        if has_swap and not has_clean:
            return 1.0, a
        if has_clean and not has_swap:
            return 0.0, a
        return 0.5, a
