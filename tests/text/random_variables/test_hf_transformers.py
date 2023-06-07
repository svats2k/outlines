import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from outlines.text.random_variables.hf_transformers import HFTransformersRV


def test_base():
    def MockModel(input_ids: torch.Tensor, attention_mask=None):
        from typing import NamedTuple

        class Output(NamedTuple):
            logits: torch.Tensor

        return Output(torch.ones(input_ids.shape[0], input_ids.shape[1] + 1, 3))

    next_token_dist = HFTransformersRV(MockModel)

    rng = torch.Generator()

    input_ids = torch.tensor([[0, 1, 2, 3]])
    next_token = next_token_dist(rng, input_ids)
    assert next_token.shape == (1,)

    next_token = next_token_dist(rng, input_ids, samples=3)
    assert next_token.shape == (3,)

    input_ids = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]])
    next_token = next_token_dist(rng, input_ids)
    assert next_token.shape == (2, 1)

    next_token = next_token_dist(rng, input_ids, samples=3)
    assert next_token.shape == (2, 3)


def test_integration():
    model_name = "hf-internal-testing/tiny-random-GPT2LMHeadModel"
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    next_token_dist = HFTransformersRV(model)

    rng = torch.Generator()
    rng.manual_seed(0)

    prompt = "test"
    tokens = tokenizer.encode(
        prompt,
        padding=True,
        add_special_tokens=False,
        return_tensors="pt",
    )
    next_token = next_token_dist(rng, tokens)
    assert next_token.shape == (1,)

    next_token = next_token_dist(rng, tokens, samples=3)
    assert next_token.shape == (3,)

    prompts = ["test1", "test2", "test3 very long"]
    output = tokenizer.batch_encode_plus(
        prompts,
        padding=True,
        add_special_tokens=False,
        return_tensors="pt",
    )

    next_token = next_token_dist(rng, output["input_ids"])
    assert next_token.shape == (3, 1)

    next_token = next_token_dist(
        rng, output["input_ids"], output["attention_mask"], samples=2
    )
    assert next_token.shape == (3, 2)
