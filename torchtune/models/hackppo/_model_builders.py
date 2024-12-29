from torchtune.models.hackppo._component_builders import llama3_1_classifier
from torchtune.modules import (
    TransformerDecoder,
)


def llama3_1_classifier_8b(n_classes: int) -> TransformerDecoder:
    """
    Builder for creating a Llama3.1 model initialized w/ the default 8b parameter values.

    Returns:
        TransformerDecoder: Instantiation of Llama3.1 8B model
    """
    return llama3_1_classifier(
        n_classes=n_classes,
        vocab_size=128_256,
        num_layers=32,
        num_heads=32,
        num_kv_heads=8,
        embed_dim=4096,
        max_seq_len=131072,
        intermediate_dim=14336,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=500_000,
    )
