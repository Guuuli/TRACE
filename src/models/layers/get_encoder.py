import torch
import logging

from .self_attention_family import AttentionLayer, TraceAttention
from .transformer_encoder_decoder import Encoder, EncoderLayer

def get_transformer_backbone(configs):
    if configs.model_name == "TraceEncoder":
        return get_trace_encoder(configs)
    ## add other encoders here
    else:
        return get_huggingface_transformer(configs)

def get_huggingface_transformer(configs):
    from transformers import T5Config, T5EncoderModel, T5Model

    if configs.getattr("randomly_initialize_backbone", False):
        model_config = T5Config.from_pretrained(configs.transformer_backbone)
        transformer_backbone = T5Model(model_config)
        logging.info(f"Initializing randomly initialized\
                        transformer from {configs.transformer_backbone}.")
    else:
        transformer_backbone = T5EncoderModel.from_pretrained(
            configs.transformer_backbone
        )
        logging.info(f"Initializing pre-trained \
                        transformer from {configs.transformer_backbone}.")

    if configs.transformer_type == "encoder_only":
        transformer_backbone = transformer_backbone.get_encoder()
    elif configs.transformer_type == "decoder_only":
        transformer_backbone = transformer_backbone.get_decoder()

    if configs.getattr("enable_gradient_checkpointing", True):
        transformer_backbone.gradient_checkpointing_enable()
        logging.info("Enabling gradient checkpointing.")

    return transformer_backbone

def get_trace_encoder(configs):
    encoder = Encoder(
        [
            EncoderLayer(
                AttentionLayer(
                    TraceAttention(
                        attention_dropout=configs.attention_dropout,
                        output_attention=configs.output_attention,
                        d_model=configs.d_model,
                        num_heads=configs.n_heads,
                        flash_attention=configs.flash_attention,
                    ),
                    configs.d_model,
                    configs.n_heads,
                ),
                d_model=configs.d_model,
                dropout=configs.dropout,
                activation=configs.activation,
            )
            for l in range(configs.e_layers)
        ],
        norm_layer=torch.nn.LayerNorm(configs.d_model),
    )
    return encoder



