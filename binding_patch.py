"""
Single-head cross-attention binding patcher for SD1.5.

Mechanism (cleaner equivalent of cache-and-inject, no cross-run latent divergence):
The pipeline always runs the CLEAN prompt. For each cross-attention (layer L, head H)
selected in the swap-set, that head's K/V are recomputed from the SWAPPED prompt's text
embedding instead of the clean one. Q always comes from the current (clean-run) latent
state, so there is no spatial divergence between a "cached" run and the patched run.

Exact brackets (this is also the correctness proof):
  * swap-set = {}        -> output is bit-identical to a vanilla clean-prompt generation.
  * swap-set = ALL cells -> every cross-attn reads the swapped text => bit-identical to a
                            vanilla SWAPPED-prompt generation (same seed / same init latent).
A single head H swapped changes ONLY head H's K/V contribution; all other heads identical.
"""
import torch
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline, DDIMScheduler
from revisions import REVISION


class Controller:
    """Global state shared by all patched cross-attention processors.

    The intervention source is chosen INDEPENDENTLY for K and V per head, via two
    boolean masks (key_masks / value_masks). This is the OV-vs-QK decomposition:
      * K from swapped text  -> changes the attention PATTERN (routing, softmax(Q.Kt)).
      * V from swapped text  -> changes the delivered CONTENT (output = pattern . V).
    The legacy helpers (set_swap_cells / set_swap_all / set_swap_layer) set BOTH masks,
    i.e. the original FULL text-source swap, so the localization brackets are unchanged.
    """
    def __init__(self):
        self.enabled = False
        self.swap_embeds = None          # [batch, seq_txt, dim], same layout as clean enc states
        self.key_masks = {}              # layer_name -> bool tensor [heads]: swap K source
        self.value_masks = {}            # layer_name -> bool tensor [heads]: swap V source
        self.layer_heads = {}            # layer_name -> int (#heads), filled at install
        # ---- optional per-head K/V capture, for the independence assertion ----
        self.capture_layer = None        # layer_name to snapshot post-intervention K/V
        self._captured = None            # dict(key=..., value=...) cpu float [heads,seq,dim]
        self._capture_done = False

    def reset(self):
        self.enabled = False
        self.key_masks = {}
        self.value_masks = {}
        # capture config (_captured/_capture_done/capture_layer) is left intact so the
        # caller can read the snapshot after generate() returns; generate() arms it.

    def _ensure(self, ln, device):
        if ln not in self.key_masks:
            self.key_masks[ln] = torch.zeros(self.layer_heads[ln], dtype=torch.bool, device=device)
            self.value_masks[ln] = torch.zeros(self.layer_heads[ln], dtype=torch.bool, device=device)

    def set_swap_cells_kv(self, cells, swap_k, swap_v, device):
        """cells: iterable of (layer_name, head_index). For each cell set the K-source
        (swap_k) and V-source (swap_v) independently. enabled stays True even when both
        are False (the BASELINE no-op path) so the plumbing is exercised identically."""
        self.key_masks = {}
        self.value_masks = {}
        for ln, h in cells:
            self._ensure(ln, device)
            if swap_k:
                self.key_masks[ln][h] = True
            if swap_v:
                self.value_masks[ln][h] = True
        self.enabled = len(self.key_masks) > 0

    def set_swap_cells(self, cells, device):
        """Legacy FULL swap: swap BOTH K and V for each cell."""
        self.set_swap_cells_kv(cells, True, True, device)

    def set_swap_all(self, device):
        self.key_masks = {ln: torch.ones(n, dtype=torch.bool, device=device)
                          for ln, n in self.layer_heads.items()}
        self.value_masks = {ln: torch.ones(n, dtype=torch.bool, device=device)
                            for ln, n in self.layer_heads.items()}
        self.enabled = True

    def set_swap_layer(self, layer_name, device):
        n = self.layer_heads[layer_name]
        self.key_masks = {layer_name: torch.ones(n, dtype=torch.bool, device=device)}
        self.value_masks = {layer_name: torch.ones(n, dtype=torch.bool, device=device)}
        self.enabled = True


class HeadSwapProcessor:
    """diffusers AttnProcessor2_0-compatible processor with per-head text-source swap."""
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

        batch_size, seq_len, _ = hidden_states.shape
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

        # ---- the intervention: K-source and V-source chosen INDEPENDENTLY per head ----
        # K swapped => that head's attention PATTERN (routing) follows the swapped text.
        # V swapped => that head's delivered CONTENT follows the swapped text.
        if is_cross and self.c.enabled and self.layer_name in self.c.key_masks:
            kmask = self.c.key_masks[self.layer_name]    # [heads] bool
            vmask = self.c.value_masks[self.layer_name]  # [heads] bool
            if kmask.any() or vmask.any():
                enc_s = self.c.swap_embeds
                if kmask.any():
                    key_s = split(attn.to_k(enc_s))
                    key = torch.where(kmask.view(1, heads, 1, 1), key_s, key)
                if vmask.any():
                    value_s = split(attn.to_v(enc_s))
                    value = torch.where(vmask.view(1, heads, 1, 1), value_s, value)
        # ----------------------------------------------------------------------------

        # ---- optional snapshot of post-intervention K/V for the independence check ----
        if is_cross and self.layer_name == self.c.capture_layer and not self.c._capture_done:
            # store the conditional (CFG-positive) batch row, all heads: [heads, seq, dim]
            self.c._captured = dict(
                key=key[-1].detach().float().cpu().clone(),
                value=value[-1].detach().float().cpu().clone())
            self.c._capture_done = True
        # ----------------------------------------------------------------------------

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False)
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


def load_pipe(model_id="sd-legacy/stable-diffusion-v1-5", device="cuda"):
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, revision=REVISION.get(model_id), torch_dtype=torch.float16, safety_checker=None)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def install(pipe, controller: Controller):
    """Replace every cross-attention (attn2) processor with a HeadSwapProcessor."""
    procs = {}
    for name, module in pipe.unet.attn_processors.items():
        if name.endswith("attn2.processor"):
            layer = name[: -len(".processor")]
            procs[name] = HeadSwapProcessor(controller, layer)
        else:
            procs[name] = module  # keep default for self-attn (attn1)
    pipe.unet.set_attn_processor(procs)
    # record head counts per cross-attn layer
    controller.layer_heads = {}
    for name, module in pipe.unet.named_modules():
        if name.endswith("attn2") and hasattr(module, "heads"):
            controller.layer_heads[name] = module.heads
    return sorted(controller.layer_heads.keys())


@torch.no_grad()
def encode(pipe, prompt, device="cuda"):
    """Return doubled [uncond; cond] text embedding matching the CFG pipeline layout."""
    pe, ne = pipe.encode_prompt(prompt, device, 1, True, negative_prompt=None)
    return torch.cat([ne, pe], dim=0)


@torch.no_grad()
def generate(pipe, controller, clean_prompt, swap_embeds, latents, cells=None,
             swap_all=False, swap_layer=None, swap_k=True, swap_v=True,
             capture_layer=None, steps=30, guidance=7.5, device="cuda"):
    """Generate the CLEAN prompt with a chosen swap configuration.

    For per-cell configs, swap_k / swap_v independently select the K-source and
    V-source (FULL=K&V, PATTERN-only=K, VALUE-only=V, BASELINE=neither). When
    capture_layer is given, the post-intervention K/V for that layer are snapshotted
    into controller._captured (readable after this call returns)."""
    controller.reset()
    controller.swap_embeds = swap_embeds
    controller.capture_layer = capture_layer
    controller._captured = None
    controller._capture_done = False
    if swap_all:
        controller.set_swap_all(device)
    elif swap_layer is not None:
        controller.set_swap_layer(swap_layer, device)
    elif cells:
        controller.set_swap_cells_kv(cells, swap_k, swap_v, device)
    img = pipe(clean_prompt, num_inference_steps=steps, guidance_scale=guidance,
               latents=latents).images[0]
    controller.reset()
    return img
