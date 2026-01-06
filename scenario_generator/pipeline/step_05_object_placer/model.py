import torch


def generate_with_model(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> str:
    if getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": prompt}]
        input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        if torch.cuda.is_available():
            input_ids = input_ids.to(model.device)
        attention_mask = (input_ids != tokenizer.pad_token_id).long()
        gen_kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        input_len = int(input_ids.shape[-1])
    else:
        enc = tokenizer(prompt, return_tensors="pt", padding=True)
        if torch.cuda.is_available():
            enc = {k: v.to(model.device) for k, v in enc.items()}
        gen_kwargs = enc
        input_len = int(enc["input_ids"].shape[-1])

    # Log token budget so we can see headroom for generation.
    model_ctx = getattr(getattr(model, "config", None), "max_position_embeddings", None)
    headroom = None if model_ctx is None else max(0, int(model_ctx) - input_len)
    allowed_new = None if headroom is None else max(0, headroom)
    print(
        f"[DEBUG] token budget: prompt={input_len}"
        + (f", model_ctx={model_ctx}" if model_ctx is not None else ", model_ctx=unknown")
        + f", requested_new={max_new_tokens}"
        + (f", max_new_before_ctx={allowed_new}" if allowed_new is not None else ""),
        flush=True,
    )

    if not do_sample:
        temperature = None
        top_p = None

    with torch.no_grad():
        out = model.generate(
            **gen_kwargs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # IMPORTANT: decode only the newly generated tokens, not the echoed prompt.
    gen_ids = out[0][input_len:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return text.strip()


__all__ = [
    "generate_with_model",
]
