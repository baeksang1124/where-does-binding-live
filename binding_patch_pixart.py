"""
Per-head cross-attention binding patcher for PixArt-Sigma (isolated-cross-attn DiT).

PixArt is the CONTROL model of the 2x2 design (backbone x attention type): it is a Transformer (DiT) like SD3.5
but uses ISOLATED cross-attention (image queries attend over a FIXED T5 text embedding),
exactly like SD1.5 UNet -- NOT the joint self-attention of SD3.5 MM-DiT. So the text K/V
are private per head and FIXED across all blocks/steps (T5 encodes once). The intervention
is therefore the SIMPLE SD1.5-style swap (no activation-patching cache): for selected
(block, head), recompute that head's text K and/or V from the SWAPPED prompt embedding.

  K swapped -> attention PATTERN (routing/QK).   V swapped -> delivered CONTENT (value/OV).

Brackets: swap-set {} -> bit-identical clean ; swap-set ALL K&V -> swapped-prompt image.

Differs from SD1.5 binding_patch.py only in: (a) T5 encode_prompt returns an attention
mask that must flow into SDPA; (b) newer diffusers Attention may carry norm_q/norm_k
(applied if present); (c) cross_attention_dim=4096, 28 blocks x 16 heads = 448 heads.
"""
import torch
import torch.nn.functional as F
from diffusers import PixArtSigmaPipeline, PixArtTransformer2DModel
from revisions import REVISION


class Controller:
    def __init__(self):
        self.enabled = False
        self.swap_embeds = None          # [batch, seq, 4096] swapped T5 embedding (CFG-doubled)
        self.key_masks = {}              # block_name -> bool[heads]
        self.value_masks = {}
        self.layer_heads = {}            # block_name -> int
        self.capture_layer = None
        self._captured = None
        self._capture_done = False

    def reset(self):
        self.enabled = False
        self.key_masks = {}
        self.value_masks = {}

    def _ensure(self, ln, device):
        if ln not in self.key_masks:
            self.key_masks[ln] = torch.zeros(self.layer_heads[ln], dtype=torch.bool, device=device)
            self.value_masks[ln] = torch.zeros(self.layer_heads[ln], dtype=torch.bool, device=device)

    def set_swap_cells_kv(self, cells, swap_k, swap_v, device):
        self.key_masks = {}
        self.value_masks = {}
        for ln, h in cells:
            self._ensure(ln, device)
            if swap_k:
                self.key_masks[ln][h] = True
            if swap_v:
                self.value_masks[ln][h] = True
        self.enabled = len(self.key_masks) > 0

    def set_swap_all(self, device):
        self.key_masks = {ln: torch.ones(n, dtype=torch.bool, device=device)
                          for ln, n in self.layer_heads.items()}
        self.value_masks = {ln: torch.ones(n, dtype=torch.bool, device=device)
                            for ln, n in self.layer_heads.items()}
        self.enabled = True


class HeadSwapProcessor:
    """AttnProcessor2_0-compatible cross-attn processor with per-head text-source swap."""
    def __init__(self, controller: Controller, layer_name: str):
        self.c = controller
        self.layer_name = layer_name

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, temb=None, **kwargs):
        is_cross = encoder_hidden_states is not None
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            b, ch, h, w = hidden_states.shape
            hidden_states = hidden_states.view(b, ch, h * w).transpose(1, 2)

        batch_size, seq_len, _ = (hidden_states.shape if encoder_hidden_states is None
                                  else (hidden_states.shape[0], hidden_states.shape[1], None))

        if attention_mask is not None:
            key_tokens = encoder_hidden_states.shape[1]
            attention_mask = attn.prepare_attention_mask(attention_mask, key_tokens, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        enc = hidden_states if encoder_hidden_states is None else encoder_hidden_states
        if attn.norm_cross:
            enc = attn.norm_encoder_hidden_states(enc)

        key = attn.to_k(enc)
        value = attn.to_v(enc)

        heads = attn.heads
        head_dim = query.shape[-1] // heads

        def split(t):
            return t.view(t.shape[0], -1, heads, head_dim).transpose(1, 2)

        query = split(query)
        key = split(key)
        value = split(value)

        # newer diffusers: optional per-head QK RMSNorm
        if getattr(attn, "norm_q", None) is not None:
            query = attn.norm_q(query)
        if getattr(attn, "norm_k", None) is not None:
            key = attn.norm_k(key)

        # ---- the intervention: per-head K/V from the swapped text (fixed embedding) ----
        if is_cross and self.c.enabled and self.layer_name in self.c.key_masks:
            kmask = self.c.key_masks[self.layer_name]
            vmask = self.c.value_masks[self.layer_name]
            if kmask.any() or vmask.any():
                enc_s = self.c.swap_embeds
                if kmask.any():
                    key_s = split(attn.to_k(enc_s))
                    if getattr(attn, "norm_k", None) is not None:
                        key_s = attn.norm_k(key_s)
                    key = torch.where(kmask.view(1, heads, 1, 1), key_s, key)
                if vmask.any():
                    value_s = split(attn.to_v(enc_s))
                    value = torch.where(vmask.view(1, heads, 1, 1), value_s, value)
        # --------------------------------------------------------------------------------

        if is_cross and self.layer_name == self.c.capture_layer and not self.c._capture_done:
            self.c._captured = dict(key=key[-1].detach().float().cpu().clone(),
                                    value=value[-1].detach().float().cpu().clone())
            self.c._capture_done = True

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(b, ch, h, w)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


def load_pipe(device="cuda", res=512):
    """512-MS transformer + T5/VAE from 1024-MS base. T5 in fp16 (encode-once)."""
    base = "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS"
    tf = PixArtTransformer2DModel.from_pretrained(
        "PixArt-alpha/PixArt-Sigma-XL-2-512-MS", subfolder="transformer",
        revision=REVISION["PixArt-alpha/PixArt-Sigma-XL-2-512-MS"], torch_dtype=torch.float16)
    pipe = PixArtSigmaPipeline.from_pretrained(base, revision=REVISION[base], transformer=tf,
                                               torch_dtype=torch.float16)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def install(pipe, controller: Controller):
    """Replace every cross-attention (attn2) processor. attn1 (self-attn) untouched."""
    procs = {}
    for name, module in pipe.transformer.attn_processors.items():
        if "attn2" in name and name.endswith(".processor"):
            procs[name] = HeadSwapProcessor(controller, name[: -len(".processor")])
        else:
            procs[name] = module
    pipe.transformer.set_attn_processor(procs)
    controller.layer_heads = {}
    for name, module in pipe.transformer.named_modules():
        if name.endswith("attn2") and hasattr(module, "heads"):
            controller.layer_heads[name] = module.heads
    return sorted(controller.layer_heads.keys())


@torch.no_grad()
def encode(pipe, prompt, device="cuda"):
    """Return CFG-doubled [uncond; cond] text embedding + attention mask.

    CRITICAL: attn2.to_k/to_v expect the 1152-dim CAPTION-PROJECTED embedding, not the
    raw 4096-dim T5 output. caption_projection is a timestep-independent MLP applied once
    in the transformer forward, so we pre-apply it here -> swap_embeds is (2,300,1152)."""
    pe, pmask, ne, nmask = pipe.encode_prompt(prompt, do_classifier_free_guidance=True,
                                              device=device)
    emb = torch.cat([ne, pe], dim=0)                       # (2,300,4096)
    mask = torch.cat([nmask, pmask], dim=0)                # (2,300)
    proj = pipe.transformer.caption_projection(emb.to(pipe.transformer.dtype))  # (2,300,1152)
    return proj, mask


@torch.no_grad()
def generate(pipe, controller, clean_prompt, swap_embeds, latents, cells=None,
             swap_all=False, swap_k=True, swap_v=True, capture_layer=None,
             steps=20, guidance=4.5, height=512, width=512, device="cuda"):
    controller.reset()
    controller.swap_embeds = swap_embeds
    controller.capture_layer = capture_layer
    controller._captured = None
    controller._capture_done = False
    if swap_all:
        controller.set_swap_all(device)
    elif cells:
        controller.set_swap_cells_kv(cells, swap_k, swap_v, device)
    img = pipe(clean_prompt, num_inference_steps=steps, guidance_scale=guidance,
               height=height, width=width, latents=latents).images[0]
    controller.reset()
    return img
