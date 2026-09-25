"""
Per-head joint-attention binding patcher for SD3.5-medium (MM-DiT).

SD3 uses JOINT self-attention: at every transformer block, the image tokens
(to_q/to_k/to_v) and the text tokens (add_q_proj/add_k_proj/add_v_proj) are
projected, concatenated along the sequence dim, attended jointly, then split.
The image queries attend over [image_K ; text_K]; the TEXT contribution to that
routing/content is exactly (text_K, text_V) = the add_k_proj/add_v_proj outputs.

Binding intervention = per (block, head) swap of the TEXT-token K and/or V from
the SWAPPED prompt instead of the clean prompt. K-swap changes the attention
PATTERN (routing/QK); V-swap changes the delivered CONTENT (value/OV). This is
the exact MM-DiT analog of the SD1.5 cross-attn swap in binding_patch.py.

KEY DIFFERENCE vs SD1.5: the text stream is NOT fixed -- it evolves across the
24 blocks and (because image tokens influence it) differs at every denoising
step. So we cannot recompute K/V from one fixed embedding. Instead this is
standard activation patching: run the swapped prompt once with capture=True to
record, per (block, denoising-step), the text stream `encoder_hidden_states`
fed into that block's joint attention; then in the clean run recompute the
swapped text K/V from the cached stream for the selected heads.

Brackets (correctness proof):
  * swap-set = {}            -> no injection -> bit-identical to clean generation.
  * swap-set = ALL cells K&V -> every block's image queries read the swapped
                                text K/V (the head masks apply to both CFG rows)
                                -> strongly reproduces the
                                swapped-prompt binding (approximate: the text
                                stream cache is stale under the new trajectory,
                                a standard activation-patching caveat).
A single (block, head) swap changes only that head's text K/V rows.
"""
import torch
import torch.nn.functional as F
from diffusers import StableDiffusion3Pipeline


class Controller:
    """Shared state for all patched joint-attention processors.

    Two modes:
      capture: each block records its incoming text stream (encoder_hidden_states)
               per denoising step -> swap_ctx[block] = [step0, step1, ...].
      patch:   for selected heads, recompute text K/V from the cached swapped
               stream (key_masks / value_masks chosen independently for the
               OV-vs-QK decomposition, exactly like the SD1.5 controller).
    """
    def __init__(self):
        self.mode = "off"               # "off" | "capture" | "patch"
        self.key_masks = {}             # block_idx -> bool[heads]: swap text-K source
        self.value_masks = {}           # block_idx -> bool[heads]: swap text-V source
        self.layer_heads = {}           # block_idx -> int (#heads), filled at install
        self.swap_ctx = {}              # block_idx -> list[Tensor]  (one per denoising step)
        self.step_ctr = {}              # block_idx -> int  running step counter this run

    # ---- run setup -------------------------------------------------------
    def begin_capture(self):
        self.mode = "capture"
        self.swap_ctx = {b: [] for b in self.layer_heads}
        self.step_ctr = {b: 0 for b in self.layer_heads}
        self.key_masks = {}
        self.value_masks = {}

    def begin_patch(self):
        self.mode = "patch"
        self.step_ctr = {b: 0 for b in self.layer_heads}

    def off(self):
        self.mode = "off"
        self.key_masks = {}
        self.value_masks = {}
        self.step_ctr = {b: 0 for b in self.layer_heads}

    # ---- mask helpers (cells = (block_idx, head_idx)) --------------------
    def _ensure(self, b, device):
        if b not in self.key_masks:
            self.key_masks[b] = torch.zeros(self.layer_heads[b], dtype=torch.bool, device=device)
            self.value_masks[b] = torch.zeros(self.layer_heads[b], dtype=torch.bool, device=device)

    def set_swap_cells_kv(self, cells, swap_k, swap_v, device):
        self.key_masks = {}
        self.value_masks = {}
        for b, h in cells:
            self._ensure(b, device)
            if swap_k:
                self.key_masks[b][h] = True
            if swap_v:
                self.value_masks[b][h] = True

    def set_swap_all(self, swap_k, swap_v, device):
        self.key_masks = {b: torch.full((n,), bool(swap_k), dtype=torch.bool, device=device)
                          for b, n in self.layer_heads.items()}
        self.value_masks = {b: torch.full((n,), bool(swap_v), dtype=torch.bool, device=device)
                            for b, n in self.layer_heads.items()}


class JointHeadSwapProcessor:
    """JointAttnProcessor2_0 replica with per-head swapped-text K/V injection."""
    def __init__(self, controller: Controller, block_idx: int):
        self.c = controller
        self.b = block_idx

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, *args, **kwargs):
        residual = hidden_states
        batch_size = hidden_states.shape[0]

        # ---- image ("sample") projections ----
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        def split(t):
            return t.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        query = split(query); key = split(key); value = split(value)
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # ---- text ("context") projections ----
        if encoder_hidden_states is not None:
            # capture the incoming text stream for this (block, step), then advance.
            if self.c.mode == "capture":
                # store on CPU: the full (step x block) cache (~1.4GB) must coexist with
                # the resident Qwen grader; injection moves the slice back to GPU on demand.
                self.c.swap_ctx[self.b].append(encoder_hidden_states.detach().to("cpu", copy=True))

            eq = split(attn.add_q_proj(encoder_hidden_states))
            ek = split(attn.add_k_proj(encoder_hidden_states))
            ev = split(attn.add_v_proj(encoder_hidden_states))
            if attn.norm_added_q is not None:
                eq = attn.norm_added_q(eq)
            if attn.norm_added_k is not None:
                ek = attn.norm_added_k(ek)

            # ---- the intervention: swap text K/V per head from the cached swapped stream ----
            if self.c.mode == "patch" and self.b in self.c.key_masks:
                kmask = self.c.key_masks[self.b]
                vmask = self.c.value_masks[self.b]
                if kmask.any() or vmask.any():
                    idx = self.c.step_ctr[self.b]
                    ehs_s = self.c.swap_ctx[self.b][idx].to(encoder_hidden_states.device)
                    if kmask.any():
                        ek_s = split(attn.add_k_proj(ehs_s))
                        if attn.norm_added_k is not None:
                            ek_s = attn.norm_added_k(ek_s)
                        ek = torch.where(kmask.view(1, attn.heads, 1, 1), ek_s, ek)
                    if vmask.any():
                        ev_s = split(attn.add_v_proj(ehs_s))
                        ev = torch.where(vmask.view(1, attn.heads, 1, 1), ev_s, ev)
            if self.c.mode == "patch":
                self.c.step_ctr[self.b] += 1
            # ------------------------------------------------------------------------------

            query = torch.cat([query, eq], dim=2)
            key = torch.cat([key, ek], dim=2)
            value = torch.cat([value, ev], dim=2)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            hidden_states, encoder_hidden_states = (
                hidden_states[:, : residual.shape[1]],
                hidden_states[:, residual.shape[1]:],
            )
            if not attn.context_pre_only:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if encoder_hidden_states is not None:
            return hidden_states, encoder_hidden_states
        return hidden_states


def load_pipe(model_id="stabilityai/stable-diffusion-3.5-medium", device="cuda"):
    pipe = StableDiffusion3Pipeline.from_pretrained(
        model_id, text_encoder_3=None, tokenizer_3=None, torch_dtype=torch.float16)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def install(pipe, controller: Controller):
    """Install a JointHeadSwapProcessor on every block's joint attention (attn).
    attn2 (image-only self-attn in the dual blocks) is left at its default."""
    tf = pipe.transformer
    controller.layer_heads = {}
    for i, blk in enumerate(tf.transformer_blocks):
        blk.attn.processor = JointHeadSwapProcessor(controller, i)
        controller.layer_heads[i] = blk.attn.heads
    return sorted(controller.layer_heads.keys())


@torch.no_grad()
def capture_swapped(pipe, controller, swapped_prompt, seed, steps=28,
                    guidance=7.0, height=512, width=512, device="cuda"):
    """Run the swapped prompt once, recording the per-(block,step) text stream
    into controller.swap_ctx. Returns the swapped-prompt PIL image (also the
    reference for the recovery denominator)."""
    controller.begin_capture()
    g = torch.Generator(device=device).manual_seed(seed)
    img = pipe(swapped_prompt, num_inference_steps=steps, guidance_scale=guidance,
               height=height, width=width, generator=g).images[0]
    controller.off()
    return img


@torch.no_grad()
def generate(pipe, controller, clean_prompt, seed, cells=None, swap_k=True, swap_v=True,
             swap_all=False, steps=28, guidance=7.0, height=512, width=512, device="cuda"):
    """Generate the CLEAN prompt, injecting swapped text K/V for the chosen cells.
    Requires controller.swap_ctx to be populated by capture_swapped with the SAME
    seed/steps/size. cells = iterable of (block_idx, head_idx)."""
    controller.begin_patch()
    if swap_all:
        controller.set_swap_all(swap_k, swap_v, device)
    elif cells:
        controller.set_swap_cells_kv(cells, swap_k, swap_v, device)
    # else: no masks -> pure clean (bracket)
    g = torch.Generator(device=device).manual_seed(seed)
    img = pipe(clean_prompt, num_inference_steps=steps, guidance_scale=guidance,
               height=height, width=width, generator=g).images[0]
    controller.off()
    return img
