import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache

MODEL = "models/draft-0.5b"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map="cuda")
model.eval()

ids = tok("The capital of France is Paris. The capital of Japan is Tokyo. "
          "The capital of Egypt is Cairo. The capital of Peru is",
          return_tensors="pt").input_ids.to("cuda")
split = ids.shape[1] - 10
head, tail = ids[:, :split], ids[:, split:]

cache = DynamicCache(config=model.config)
print("is_croppable:", getattr(cache, "is_croppable", "ATTR MISSING"))

with torch.inference_mode():
    model(input_ids=head, past_key_values=cache, use_cache=True)
    print("after head:", cache.get_seq_length(), "expected:", split)

    ref = model(input_ids=tail, past_key_values=cache, use_cache=True).logits
    print("after tail:", cache.get_seq_length(), "expected:", ids.shape[1])

    cache.crop(split)
    print("after crop:", cache.get_seq_length(), "expected:", split)

    again = model(input_ids=tail, past_key_values=cache, use_cache=True).logits

maxdiff = (ref - again).abs().max().item()
print("max logit diff:", maxdiff)
print("VERDICT:", "PASS" if torch.allclose(ref, again, atol=1e-3) else "FAIL — crop corrupts state")