import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from src.models.timeseries_encoders.ts_encoder import TS_Encoder
from src.models.timeseries_encoders.base import BaseModel
from src.data.base import TimeseriesOutputs
from src.utils.masking import Masking
from src.models.layers.self_attention_family import ResidualCrossAttention
from src.models.timeseries_encoders.tsfm import UnifiedTimeSeriesModel


class MultiModalEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.device = torch.device(f"cuda:{self.args.rank}" if torch.cuda.is_available() else "cpu")
        
        if hasattr(args, "model_name") and args.model_name.lower() in ["moment", "time-moe", "timer", "chronos"]:
            self._load_tsfm()
            self.ts_embedding_dim = self.tsfm.embedding_dim
        else:
            self._load_model("pretraining")
            self.ts_embedding_dim = args.d_model
        
        # Get the embedding dimension
        self.cross_attend = args.cross_attend
        self.text_embedding_dim = self.get_text_encoder_dimension(args.text_encoder_name)
        self.text_adapter = nn.Linear(self.text_embedding_dim, self.ts_embedding_dim)
        self.text_adapter.weight.data.zero_()
        self.text_adapter.bias.data.zero_()
        for i in range(min(self.text_embedding_dim, self.ts_embedding_dim)):
            self.text_adapter.weight.data[i, i] = 1.0
        
        self.norm = nn.LayerNorm(self.ts_embedding_dim)
        self.dropout = nn.Dropout(0.1)

        if self.cross_attend:
            self.channel_cross_attn = ResidualCrossAttention(d_model=self.ts_embedding_dim, n_heads=4, dropout=0.1, use_layernorm=True)
            
        self.to(self.device)
    def _load_model(self, pretraining_task_name):
        pretraining_args = deepcopy(self.args)
        pretraining_args.task_name = pretraining_task_name

        checkpoint = BaseModel.load_pretrained_weights(
            run_name=pretraining_args.pretraining_run_name,
            opt_steps=pretraining_args.pretraining_opt_steps,
            model_name=pretraining_args.model_name,
        )
        if self.args.model_name in ["TraceEncoder"]:
            self.ts_encoder = TS_Encoder(configs=pretraining_args)
        else:
            raise NotImplementedError(f"Model {self.args.model_name} not implemented for pretraining")
        
        new_state_dict = {}
        for k, v in checkpoint["model_state_dict"].items():
            if k.startswith("module."):
                new_state_dict[k[len("module."):]] = v
            else:
                new_state_dict[k] = v
        self.ts_encoder.load_state_dict(new_state_dict)
        
        if self.args.finetuning_mode == "linear-probing":
            for name, param in self.ts_encoder.named_parameters():
                if not name.startswith("head"):
                    param.requires_grad = False
        elif self.args.finetuning_mode == "end-to-end":
            pass
        else:
            raise NotImplementedError(
                f"Finetuning mode {self.args.finetuning_mode} not implemented"
            )

        print("====== Frozen parameter status ======")
        for name, param in self.ts_encoder.named_parameters():
            if param.requires_grad:
                print("Not frozen:", name)
            else:
                print("Frozen:", name)
        print("=====================================")
        return 
        
    def _load_tsfm(self):
        args = deepcopy(self.args)
        args.num_classes = self.args.num_class
        self.tsfm = UnifiedTimeSeriesModel(args)
        
        if self.args.model_name == "chronos":
            for param in self.tsfm.backbone.model.parameters():
                param.requires_grad = True
        else:
            for name, param in self.tsfm.backbone.named_parameters():
                param.requires_grad = True

    def get_text_encoder_dimension(self, text_encoder_name):
        model_dimensions = {
            "sentence-transformers/all-mpnet-base-v2": 768,
            "sentence-transformers/all-MiniLM-L6-v2": 384,
            "bert-base-uncased": 768,
            "bert-large-uncased": 1024,
            "roberta-base": 768,
            "roberta-large": 1024,
            "distilbert-base-uncased": 768,
            "distilroberta-base": 768,
            "albert-base-v2": 768,
            "albert-large-v2": 1024,
            "nomic-ai/nomic-embed-text-v1.5": 768,
            "BAAI/bge-base-en-v1.5": 768,
            "BAAI/bge-large-en-v1.5": 1024,
            "BAAI/bge-m3": 1024,
            "Alibaba-NLP/gte-Qwen2-1.5B-instruct": 1536,
        }
        return model_dimensions[text_encoder_name]
    
    def forward(
        self,
        x_enc: torch.Tensor,
        pretrain_mask: torch.Tensor = None,
        input_mask: torch.Tensor = None,
        channel_description_emb: torch.Tensor = None,
        description_emb: torch.Tensor = None,
        event_emb: torch.Tensor = None,
        **kwargs
        ):
        if hasattr(self.args, "model_name") and self.args.model_name.lower() in ["moment", "time-moe", "timer", "chronos"]:
            return self._tsfm_forward(x_enc, description_emb, event_emb)
        else:
            if pretrain_mask is None:
                pretrain_mask = self.ts_encoder.mask_generator.generate_mask(x=x_enc, input_mask=input_mask)
            pretrain_mask = pretrain_mask.to(x_enc.device)  # mask: [B, C, L]
        
            enc_out, attns = self.ts_encoder._get_encoding_out(x_enc, pretrain_mask, input_mask)
            
            # Decoder
            input_mask_patch_view = Masking.convert_seq_to_patch_view(input_mask, self.ts_encoder.patch_len)
            dec_out = self.ts_encoder.head["reconstruct_head"](enc_out, shape=self.ts_encoder.dec_shape)  # [B, C, L]
            class_out = self.ts_encoder.head["classification_head"](enc_out, input_mask_patch_view, shape=self.ts_encoder.dec_shape)  # [B, n_classes]
            # De-Normalization
            dec_out = self.ts_encoder.normalizer(x=dec_out, mode="denorm")
            emb_dict = self.ts_encoder.embedding_head(enc_out, input_mask_patch_view, shape=self.ts_encoder.dec_shape)

            if description_emb.shape[-1] != self.ts_embedding_dim:
                description_emb = self.text_adapter(description_emb)  #[B, d_model]
                event_emb = self.text_adapter(event_emb)  #[B, d_model]
                channel_description_emb = self.text_adapter(channel_description_emb)  #[B, C, d_model]
                B,C,D = channel_description_emb.shape
                
                description_emb = self.norm(description_emb)
                description_emb = self.dropout(description_emb)
                description_emb = F.normalize(description_emb, dim=-1)  # L2 norm
                
                event_emb = self.norm(event_emb)
                event_emb = self.dropout(event_emb)
                event_emb = F.normalize(event_emb, dim=-1)  # L2 norm
                
                channel_description_emb = self.norm(channel_description_emb.reshape(-1, self.ts_embedding_dim))
                channel_description_emb = self.dropout(channel_description_emb)
                channel_description_emb = F.normalize(channel_description_emb, dim=-1)  # L2 norm
                channel_description_emb = channel_description_emb.reshape(B, C, self.ts_embedding_dim)
            
            ts_embeddings = emb_dict["global"]
            cls_embedding = emb_dict["cls"]
            channel_embeddings = emb_dict["channels"]
            
            if self.cross_attend:
                channel_description_emb = self.channel_cross_attn(channel_description_emb, channel_embeddings)
                channel_embeddings = self.channel_cross_attn(channel_embeddings, channel_description_emb)
            
            
            
            return TimeseriesOutputs(
                    input_mask=input_mask,  # [B, C, L]
                    reconstruction=dec_out,  # [B, C, L]
                    pretrain_mask=pretrain_mask,  # [B, C, L]   
                    classification=class_out,  # [B, n_classes]
                    embeddings=ts_embeddings, # [B, d_model]
                    channel_embeddings=channel_embeddings, # [B, C, d_model]
                    cls_embedding=cls_embedding, # [B, d_model]
                    description_emb=description_emb.float(), # [B, d_model]
                    event_emb=event_emb.float(), # [B, d_model]
                    channel_description_emb=channel_description_emb.float(), # [B, C, d_model]
                )
    
    
    def _tsfm_forward(self, x_enc, description_emb, event_emb):
        
        ts_embeddings = self.tsfm.get_embedding(x_enc)
        class_out = self.tsfm.classify(x_enc)
        if description_emb.shape[-1] != self.ts_embedding_dim:
            description_emb = self.text_adapter(description_emb)  #[B, d_model]
            event_emb = self.text_adapter(event_emb)  #[B, d_model]
            
            description_emb = self.norm(description_emb)
            description_emb = self.dropout(description_emb)
            description_emb = F.normalize(description_emb, dim=-1)  # L2 norm
            
            event_emb = self.norm(event_emb)
            event_emb = self.dropout(event_emb)
            event_emb = F.normalize(event_emb, dim=-1)  # L2 norm
    
        return TimeseriesOutputs(
            embeddings=ts_embeddings,
            classification=class_out,
            description_emb=description_emb.float(),
            event_emb=event_emb.float(),
        )
        
    def get_ts_embedding(self, x_enc, input_mask):
        assert hasattr(self.args, "model_name") and self.args.model_name == "TraceEncoder"
        pretrain_mask = torch.ones_like(input_mask)
        
        enc_out, attns = self.ts_encoder._get_encoding_out(x_enc, pretrain_mask, input_mask)
        input_mask_patch_view = Masking.convert_seq_to_patch_view(input_mask, self.ts_encoder.patch_len)
        emb_dict = self.ts_encoder.embedding_head(enc_out, input_mask_patch_view, shape=self.ts_encoder.dec_shape)
        
        return TimeseriesOutputs(
            input_mask=input_mask,
            embeddings=emb_dict["global"], # [B, d_model]
            channel_embeddings=emb_dict["channels"], # [B, C, d_model]
            cls_embedding=emb_dict["cls"], # [B, d_model]
        )