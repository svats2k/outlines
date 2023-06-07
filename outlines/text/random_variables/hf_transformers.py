from typing import Optional

import torch
from transformers import PreTrainedModel


class HFTransformersRV:
    """Represents a token random variable defined by a `transformers` model.

    Attributes
    ----------
    model
        A model compatible with the `transformers` API.

    """

    def __init__(self, model: PreTrainedModel):
        self.model = model

    def __call__(
        self,
        rng: torch.Generator,
        token_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        samples: int = 1,
    ):
        """Generate a new token id given a tensor of token ids.

        Parameters
        ----------
        rng
            An object that manages the state of Pytorch's random number
            generator.
        token_ids
            The tokenized prompt.
        attention_mask
        samples
            The number of token ids to samples.

        """
        output = self.model(token_ids, attention_mask=attention_mask)
        next_token_logits = output.logits[:, -1, :]
        probs = torch.nn.functional.softmax(next_token_logits, dim=-1).squeeze()
        next_token_ids = torch.multinomial(probs, num_samples=samples, generator=rng)

        return next_token_ids
