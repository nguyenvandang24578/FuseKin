import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from typing import Any, Optional, Tuple, Type
from .common import MLPBlock, LayerNorm2d

class PromptEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        kpt_prompt: bool = False,
        mask_prompt: bool = False,
        seq_len: int = 16,
    ) -> None:
        """
        Encodes prompts for input to SAM's mask decoder.

        Arguments:
          embed_dim (int): The prompts' embedding dimension
          image_embedding_size (tuple(int, int)): The spatial size of the
            image embedding, as (H, W).
          input_image_size (int): The padded size of the image as input
            to the image encoder, as (H, W).
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.kpt_mlp = nn.Sequential(
            nn.Linear(2, embed_dim // 2),   # x, y
            nn.LayerNorm(embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim)
        )    
        self.temporal_conv_kp = nn.Conv1d(
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=3,
            padding=1,  # padding=1 để giữ nguyên độ dài chuỗi T
            groups=embed_dim, # Depthwise Conv: tối ưu tham số và tốc độ
            bias=False
        )
        self.temporal_norm_kp = nn.LayerNorm(embed_dim)
        self.seq_len = seq_len

        if kpt_prompt:
            self.num_kpts: int = 19   #
            self.kpt_embeddings = nn.Embedding(self.num_kpts, embed_dim)
            self.not_kpt_embeddings = nn.Embedding(1, embed_dim)
            self.use_kpt_prompt = True
        else:
            self.use_kpt_prompt = False

        # For Mask prompt
        if mask_prompt:
            mask_in_chans = embed_dim
            self.mask_encoder = nn.Sequential(
                # Input: 224x224 -> 56x56
                nn.Conv2d(1, mask_in_chans // 4, kernel_size=4, stride=4),
                LayerNorm2d(mask_in_chans // 4),
                nn.GELU(),
                
                # 56x56 -> 14x14
                nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=4, stride=4),
                LayerNorm2d(mask_in_chans),
                nn.GELU(),
                
                # CHÍNH SỬA: 14x14 -> 1x1 (Nén toàn bộ không gian)
                nn.AdaptiveAvgPool2d(1),
            )
            self.no_mask_embed = nn.Embedding(1, mask_in_chans)
            self.use_mask_prompt = True
        else:
            self.use_mask_prompt = False

    def _embed_kpts(self, kpts: torch.Tensor) -> torch.Tensor:
        """
        Args:
            kpts: (B*T, 19, 3) - [x, y, conf] normalized về [0,1]
        Returns:
            (B*T, 19, embed_dim)
        """
        kpts = kpts.to(self._get_device())
        bs = kpts.shape[0] // self.seq_len
        num_kp = self.num_kpts
        loc  = kpts        
        # MLP encode vị trí → (B*T, 19, embed_dim)
        x = self.kpt_mlp(loc)
        x_reshaped = x.view(bs, self.seq_len, num_kp, -1) # -> (B, T, num_kp, dim)
        x_conv_in = x_reshaped.permute(0, 2, 3, 1) # -> (B, 2, dim, T)
        x_t = x_conv_in.reshape(bs * num_kp, -1, self.seq_len) # -> (B * 2, dim, T)
        # Bước 2: Bắt chuyển động qua trục thời gian T
        x_t = self.temporal_conv_kp(x_t)          # Trượt Conv1d dọc theo T

        x_restored = x_t.reshape(bs, num_kp, -1, self.seq_len) # -> (B, 2, dim, T)
        x_restored = x_restored.permute(0, 3, 1, 2).contiguous() # -> (B, T, 2, dim)        
        x_restored = x_restored.reshape(bs * self.seq_len, num_kp, -1) # -> (B * T, 2, dim)
        kpt_embedding = self.temporal_norm_kp(x + x_restored)

        # Thêm learned type embedding cho từng keypoint
        kpt_embedding = kpt_embedding + self.kpt_embeddings.weight  # (num_kp, embed_dim)
        
        return kpt_embedding  # (B*T, num_kp, embed_dim)
    

    def _embed_masks(self, masks: torch.Tensor) -> torch.Tensor:
        device = masks.device
        
        # 1. Xử lý 5 chiều (Video)
        if masks.dim() == 5:
            B, S, C, H, W = masks.shape
            masks = masks.view(B * S, C, H, W) # Gộp Batch & Seq thành (B*S, 1, 224, 224)
        else:
            B, C, H, W = masks.shape
            S = 1

        # 3. Chạy qua CNN Downscaling (Sử dụng version mask_encoder có GAP)
        # Input: (B*S, 1, 224, 224) -> Output: (B*S, 512, 1, 1)
    # Chạy qua encoder: output là (B*T, 512, 1, 1)
        mask_embedding = self.mask_encoder(masks)      
        
        # Cuối cùng đưa về (B, T, 512)
        return mask_embedding.flatten(1).view(B, S, -1)
    def _get_device(self) -> torch.device:
        return self.kpt_embeddings.weight.device


    def forward(
        self,
        kpts: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            boxes: (B*T, 4) - [x, y, w, h]
            kpts: (B*T, 19, 3) - [[x, y, conf], ...]
            masks: (B, S, 1, H, W) - binary masks
            
        Returns:
            sparse_embeddings: (B*T, N, embed_dim) - N = 2 (box) + 19 (kpts)
            dense_embeddings: (B, S, embed_dim) hoặc None
        """
        # --- 1. Xác định chuẩn xác B (Batch) và BT (Batch * Time) ---
        if kpts is not None:
            BT = kpts.shape[0]
            B = BT // self.seq_len
        elif masks is not None:
            B = masks.shape[0]  # Mask đầu vào là 5 chiều
            BT = B * self.seq_len
        else:
            raise ValueError("At least one prompt must be provided!")
        
        sparse_embeddings = torch.empty(
            (BT, 0, self.embed_dim), 
            device=self._get_device()
        )

        # ===== KEYPOINT EMBEDDINGS =====
        if self.use_kpt_prompt:
            if kpts is not None:
                kpts_embeddings = self._embed_kpts(kpts)  # (BT, 19, embed_dim)
            else:
                kpts_embeddings = self.not_kpt_embeddings.weight.unsqueeze(0)
                kpts_embeddings = kpts_embeddings.expand(BT, self.num_kpts, -1)

            sparse_embeddings = torch.cat([sparse_embeddings, kpts_embeddings], dim=1) 

        # ĐẦU RA MONG MUỐN CỦA BẠN TRƯỚC KHI CỘNG LEARNABLE TOKENS
        # sparse_embeddings hiện tại đang là (B*T, N_tokens, dim)

        # ===== MASK EMBEDDINGS (Dense) =====
        if masks is not None and self.use_mask_prompt:
            dense_embeddings = self._embed_masks(masks) # Trả về (B, S, dim)
        elif self.use_mask_prompt:
            # Dùng B thay vì BT để không bị lỗi chiều
            dense_embeddings = self.no_mask_embed.weight \
                .unsqueeze(0).expand(B, self.seq_len, -1) 
        else:
            dense_embeddings = None
    
        return sparse_embeddings, dense_embeddings

