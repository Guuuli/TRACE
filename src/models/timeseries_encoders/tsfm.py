import torch
import torch.nn as nn
from typing import List


class UnifiedTimeSeriesModel(nn.Module):
    """
    A unified interface for time series foundation models.
    """
    def __init__(self, args):
        """
        Initialize the unified time series model.
        """
        super().__init__()
        self.args = args
        self.model_name = args.model_name.lower()
        self.check_model_name()
        self.variant = args.variant.lower() 
        self.num_classes = args.num_classes
        if hasattr(args, "rank"):
            self.device = torch.device(f"cuda:{self.args.rank}" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(f"cuda:{self.args.gpu_id}" if torch.cuda.is_available() else "cpu")
        
        # Load the appropriate model
        self.backbone = self._load_model()
            
        # Get the embedding dimension
        self.embedding_dim = self._get_embedding_dim()
        print(f"Embedding dimension: {self.embedding_dim}")
        
        # Create classification head if num_classes is provided
        if self.num_classes is not None:
            self.classification_head = self._create_classification_head([256,128,64,32])
        else:
            self.classification_head = None
            
        self.to(self.device)

    def check_model_name(self):
        if self.model_name not in ["moment", "time-moe", "chronos", "timer"]:
            raise ValueError(f"Unknown model: {self.model_name}")

    def _get_embedding_dim(self) -> int:
        """Determine the embedding dimension of the model."""
        if self.model_name == "moment":
            model_size = {"small":512, "base":768, "large":1024}
            return model_size[self.variant]  # Default for Moment
        elif self.model_name == "time-moe":
            model_size = {"small":384, "large":768}
            return model_size[self.variant]  # Default for Time-MoE
        elif self.model_name == "chronos": # not support finetuning
            model_size = {"tiny":256, "mini":384, "small":512, "base":768, "large":1024}
            return model_size[self.variant]  # Default for Chronos
        elif self.model_name == "timer":
            return 1024  
        else:
            raise ValueError(f"Unknown embedding dimension for model {self.model_name} with variant {self.variant}")

    def _create_classification_head(self, hidden_dims: List[int]) -> nn.Module:
        """
        Create a classification head with the given hidden dimensions.
        
        Args:
            hidden_dims: List of hidden dimensions for the MLP
            
        Returns:
            An MLP module for classification
        """
        layers = []
        input_dim = self.embedding_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            input_dim = hidden_dim
            
        layers.append(nn.Linear(input_dim, self.num_classes))
        
        return nn.Sequential(*layers)

    def _load_model(self) -> nn.Module:
        """
        Load the appropriate model based on model_name.
        
        Returns:
            The loaded model backbone
        """
        if self.model_name == "moment":
            return self._load_moment()
        elif self.model_name == "time-moe":
            return self._load_time_moe()
        elif self.model_name == "chronos":
            return self._load_chronos()
        elif self.model_name == "timer":
            return self._load_timer()
        else:
            raise ValueError(f"Unknown model: {self.model_name}")

    def _load_moment(self) -> nn.Module:
        """Load Moment model."""
        try:
            from momentfm import MOMENTPipeline
            
            # Map variant to model size
            model_size = ["small", "base", "large"]
            
            if self.variant not in model_size:
                raise ValueError(f"Invalid variant {self.variant} for Moment. Choose from {model_size}")
                
            print(f"Loading Moment {self.variant} model")
            
            model = MOMENTPipeline.from_pretrained(f"AutonLab/MOMENT-1-{self.variant}", 
                                                   model_kwargs={"task_name": "embedding"}
                                                   ).to(self.device)
            model.init()
            model.eval()
            for param in model.parameters():
                param.requires_grad = False
            return model
            
        except ImportError:
            raise ImportError("Failed to import Moment.")

    def _load_time_moe(self) -> nn.Module:
        """Load Time-MoE model."""
        try:
            from transformers import AutoModelForCausalLM
            device_str = f"{self.device.type}:{self.device.index}" if self.device.type == "cuda" else "cpu"
            # Map variant to model size
            model_size = {"small": "50M", "large": "200M"}
            
            if self.variant not in model_size:
                raise ValueError(f"Invalid variant {self.variant} for Time-MoE. Choose from {list(model_size.keys())}")
                
            print(f"Loading Time-MoE {model_size[self.variant]} model")
            
            model = AutoModelForCausalLM.from_pretrained(f'Maple728/TimeMoE-{model_size[self.variant]}',
                                                         device_map={"": device_str},
                                                         trust_remote_code=True
                                                         ).to(self.device)
            model.eval()
            for param in model.parameters():
                param.requires_grad = False
            return model
            
        except ImportError:
            raise ImportError("Failed to import Time-MoE.")

    def _load_timer(self) -> nn.Module:
        """Load Timer model."""
        try:
            from transformers import AutoModelForCausalLM
            device_str = f"{self.device.type}:{self.device.index}" if self.device.type == "cuda" else "cpu"
            print("Loading Timer-XL 84M  model")
            model = AutoModelForCausalLM.from_pretrained('thuml/timer-base-84m', device_map={"": device_str}, trust_remote_code=True)
            model.eval()
            for param in model.parameters():
                param.requires_grad = False
            return model
            
        except ImportError:
            raise ImportError("Failed to import Timer-XL")


    def _load_chronos(self) -> nn.Module:
        """Load Chronos model."""
        try:
            from chronos import ChronosPipeline
            device_str = f"{self.device.type}:{self.device.index}" if self.device.type == "cuda" else "cpu"
            model_size = ["tiny", "mini", "small", "base", "large"]
            if self.variant not in model_size:
                raise ValueError(f"Invalid variant {self.variant} for Chronos. Choose from {model_size}")
            print(f"Loading Chronos {self.variant} model")
            
            pipeline = ChronosPipeline.from_pretrained(f"amazon/chronos-t5-{self.variant}",
                                                    device_map={"": device_str}, 
                                                    torch_dtype=torch.bfloat16)
            pipeline.model.eval()
            for param in pipeline.model.parameters():
                param.requires_grad = False
            return pipeline
            
        except ImportError:
            raise ImportError("Failed to import Chronos.")
            
    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get the embedding from the model backbone. This function handles the model-specific
        embedding extraction.
        
        Args:
            x: Input tensor of shape [B,C,L] where:
               B = batch size
               C = number of channels/features
               L = sequence length
        
        Returns:
            Embedding tensor of shape [B, d_model] where D is the embedding dimension
        """
        B,C,L = x.shape
        with torch.no_grad():
            if self.model_name == "moment":
                x = x.to(self.device)
                output = self.backbone(x_enc=x)
                embedding = output.embeddings
                
            elif self.model_name == "time-moe":
                x = x.to(self.device)
                x = x.reshape(-1, x.shape[-1])  # [B*C, L]
                output = self.backbone(x, output_hidden_states=True)
                embedding = output.hidden_states[-1] # [B*C, L, d_model]
                embedding = embedding.mean(dim=1).reshape(B, C,-1) # [B, C, d_model]
                embedding = embedding.mean(dim=1) # [B, d_model]
            elif self.model_name == "timer":
                x = x.to(self.device)
                x = x.reshape(-1, x.shape[-1])  # [B*C, L]
                output = self.backbone(x, output_hidden_states=True)
                embedding = output.hidden_states[-1].squeeze(1) # [B*C, d_model]
                embedding = embedding.reshape(B,C,-1) # [B,C, d_model]
                embedding = embedding.mean(dim=1) # [B, d_model]
            elif self.model_name == "chronos":
                x = x.reshape(-1, x.shape[-1])  # [B*C, L]
                embedding, _ = self.backbone.embed(x)   #[B*C, L+1, d_model]
                embedding = embedding.mean(dim=1).to(self.device) # [B*C, d_model]
                embedding = embedding.reshape(B,C,-1) # [B,C, d_model]
                embedding = embedding.mean(dim=1).float() # [B, d_model]
            else:
                raise ValueError(f"Embedding extraction not implemented for {self.model_name}")
        assert embedding.shape == (B, self.embedding_dim)
        return embedding


    def classify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get classification logits for a time series.
        Args:
            x: Input tensor of shape [B, T, C]
        Returns:
            Logits tensor of shape [B, num_classes]
        """
        if self.classification_head is None:
            raise ValueError("Classification head not initialized. Please specify num_classes when creating the model.")
            
        embedding = self.get_embedding(x)
        logits = self.classification_head(embedding)
        return logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classify(x)
    
    