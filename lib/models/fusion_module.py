import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from einops import rearrange

# ==============================================================================
# PHẦN 1: CÁC MODULE BỔ TRỢ
# ==============================================================================

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

# ==============================================================================
# PHẦN 2: CÁC KHỐI FUSION CỐT LÕI (CFCer Components)
# ==============================================================================

class AttentionNet(nn.Module):
    """
    Module Cross-Attention (Mutual Attention) chuẩn theo lý thuyết CFCer.
    
    Logic: Dùng modal 'peer' để định hướng sự chú ý (Query), 
           nhưng lấy thông tin thực tế từ modal 'curr' (Value).
    Example: Dùng Depth soi xem tay ở đâu, để lấy feature RGB chỗ cái tay đó.
    """
    def __init__(self, dim=512, heads=8, dim_head=64, mlp_dim=768, dropout=0.1, knn_attention=True, topk=0.9):
        super(AttentionNet, self).__init__()
        self.knn_attention = knn_attention
        self.topk = topk
        self.heads = heads
        self.scale = dim_head ** -0.5

        inner_dim = dim_head * heads
        self.q = nn.Linear(dim, inner_dim, bias=False)
        self.k = nn.Linear(dim, inner_dim, bias=False)
        self.v = nn.Linear(dim, inner_dim, bias=False)
        self.norm = nn.LayerNorm(dim)
        
        # FeedForward block sau attention
        self.ffn = PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
    
    def forward(self, x_curr, x_peer):
        """
        x_curr: Modal hiện tại cần được bổ sung (Ví dụ: RGB) -> Cung cấp Key, Value
        x_peer: Modal đối tác dùng để tham chiếu (Ví dụ: Depth) -> Cung cấp Query (Đèn pin)
        """
        b, n, c, h = *x_curr.shape, self.heads

        # --- [FIX LOGIC HERE] ---
        # Query: Lấy từ Peer (Đèn pin soi đường)
        q = self.q(x_peer)
        
        # Key & Value: Lấy từ Current (Bản đồ & Kho báu)
        k = self.k(x_curr)
        v = self.v(x_curr) 

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), [q, k, v])
        
        # Tính attention score: Peer soi vào Current xem chỗ nào khớp
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        if self.knn_attention:
            mask = torch.zeros(b, self.heads, n, n, device=x_curr.device, requires_grad=False)
            # Chỉ giữ lại top-k tương đồng nhất để tránh nhiễu
            index = torch.topk(dots, k=int(dots.size(-1)*self.topk), dim=-1, largest=True)[1]
            mask.scatter_(-1, index, 1.)
            dots = torch.where(mask > 0, dots, torch.full_like(dots, float('-inf')))

        attn = dots.softmax(dim=-1)
        
        # Aggregation: Lấy thông tin từ Current (v) dựa trên sự chỉ dẫn của Peer (attn)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        # Residual connection & Norm: Cộng vào feature gốc của Current
        out = self.norm(out) + x_curr
        out = self.ffn(out) + out
        return out

class EnhanceModule(nn.Module):
    """
    Module Gating (Phần bên trái của hình vẽ CFCer).
    Tính toán trọng số để điều chỉnh cường độ feature trước khi vào Attention.
    """
    def __init__(self, dim=512):
        super(EnhanceModule, self).__init__()
        self.mlp_rgb = nn.Sequential(
            nn.Linear(dim*2, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )
        self.mlp_depth = nn.Sequential(
            nn.Linear(dim*2, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )
        self.norm = nn.LayerNorm(dim*2)
        
    def forward(self, xr, xd):
        # Concatenate để tính mối tương quan
        joint_feature = self.norm(torch.cat((xr, xd), dim=-1))
        
        # Tính điểm số quan trọng (Gating Score)
        score_rgb = self.mlp_rgb(joint_feature)
        score_depth = self.mlp_depth(joint_feature)
        
        # Nhân lại vào feature gốc
        xr = xr * score_rgb
        xd = xd * score_depth

        return xr, xd

class ComplementSpatial(nn.Module):
    """
    [QUAN TRỌNG] Đây chính là khối CFCer hoàn chỉnh.
    Kết hợp EnhanceModule và 2 nhánh AttentionNet.
    """
    def __init__(self, depths=2, dim=512):
        super(ComplementSpatial, self).__init__()

        self.att_nets = nn.ModuleList([])
        self.norm_r = nn.LayerNorm(dim)   # nhận thẳng input dim chiều
        self.norm_d = nn.LayerNorm(dim)   # nhận thẳng input dim chiều
        for _ in range(depths):
            self.att_nets.append(nn.ModuleList([
                EnhanceModule(dim),
                AttentionNet(dim), # Nhánh 1: Cải thiện RGB (dùng Depth soi)
                AttentionNet(dim)  # Nhánh 2: Cải thiện Depth (dùng RGB soi)
            ]))
    
    def forward(self, xr, xd):
        # Normalize đầu vào
        xr = self.norm_r(xr)   # (B, N, dim) -> (B, N, dim), norm theo đúng thống kê riêng của xr
        xd = self.norm_d(xd)   # (B, N, dim) -> (B, N, dim), norm theo đúng thống kê riêng của xd               
        for EM, ANM, ANK in self.att_nets:
            # 1. Enhance (Gating)
            xr, xd = EM(xr, xd)
            
            # 2. Cross Attention (Logic B: Correct Mutual Attention)
            # ANM nhiệm vụ: Update xr -> Gọi ANM(xr, xd)
            # Bên trong AttentionNet: curr=xr, peer=xd -> Q lấy từ xd, K,V lấy từ xr
            cm = ANM(xr, xd) 
            
            # ANK nhiệm vụ: Update xd -> Gọi ANK(xd, xr)
            # Bên trong AttentionNet: curr=xd, peer=xr -> Q lấy từ xr, K,V lấy từ xd
            ck = ANK(xd, xr) 
            
            xr, xd = cm, ck
        
        return xr, xd

class ComplementTemporal(nn.Module):
    """
    Module xử lý chuỗi thời gian (hoặc chuỗi View) sau khi đã fusion.
    """
    def __init__(self, depths=2, dim=512):
        super(ComplementTemporal, self).__init__()

        self.att_nets = nn.ModuleList([])
        self.norm_r = nn.LayerNorm(dim)   # nhận thẳng input dim chiều
        self.norm_d = nn.LayerNorm(dim)   # nhận thẳng input dim chiều
        for _ in range(depths):
            self.att_nets.append(nn.ModuleList([
                AttentionNet(dim),
                AttentionNet(dim)
            ]))
        
    def forward(self, xr, xd):
        # Normalize đầu vào
        xr = self.norm_r(xr)   # (B, N, dim) -> (B, N, dim), norm theo đúng thống kê riêng của xr
        xd = self.norm_d(xd)   # (B, N, dim) -> (B, N, dim), norm theo đúng thống kê riêng của xd               
        
        for ANM, ANK in self.att_nets:
            cm = ANM(xr, xd)
            ck = ANK(xd, xr)
            xr, xd = cm, ck
        return xr, xd

# ==============================================================================
# PHẦN 3: CROSS FUSION NET (Wrapper Module)
# ==============================================================================

class CrossFusionNet(nn.Module):
    def __init__(self, args, num_classes, pretrained=None):
        super(CrossFusionNet, self).__init__()
        
        # Config mặc định nếu args không có
        scc_depth = getattr(args, 'scc_depth', 2)
        tcc_depth = getattr(args, 'tcc_depth', 2)
        
        # Các module thành phần
        self.SCC_Module = ComplementSpatial(depths=scc_depth, dim=512)
        self.temp_enhance_module = EnhanceModule(dim=512)
        self.TimesFormer = ComplementTemporal(depths=tcc_depth, dim=512)

    def forward(self, hidden_feature):
        """
        Input: Tuple (seq_rgb, seq_depth, ...) 
        """
        # Unpack linh hoạt
        if len(hidden_feature) == 4:
             spatial_M, spatial_K, temporal_M, temporal_K = hidden_feature
        else:
             spatial_M, spatial_K = hidden_feature[0], hidden_feature[1]
             temporal_M, temporal_K = spatial_M, spatial_K # Fallback

        # 1. Spatial Complement (CFCer)
        comple_features_M, comple_features_K = self.SCC_Module(spatial_M, spatial_K)
        
        # 2. Temporal Enhance
        temporal_enhance_M, temporal_enhance_K = self.temp_enhance_module(temporal_M, temporal_K)
        
        # Ở đây ta đơn giản hóa luồng dữ liệu cho mô hình 3-View
        # Coi output của CFCer là input cho Temporal Fusion luôn
        x_rgb = comple_features_M
        x_depth = comple_features_K
        
        # 3. Temporal/View Fusion
        x_rgb, x_depth = self.TimesFormer(x_rgb, x_depth)
        
        # Return features để Classifier bên ngoài xử lý
        return (x_rgb, x_depth), None