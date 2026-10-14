from collections import OrderedDict
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
import clip
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==================== 多尺度融合 ====================
class MultiScaleFusion(nn.Module):
    def __init__(self, embed_dim=768, compress_dim=256, num_scales=3, max_scale=0.5):
        super().__init__()
        self.max_scale = max_scale

        def build_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, bias=False),
                nn.GroupNorm(32, out_c),
                nn.ReLU(inplace=True)
            )

        self.compress_layers = nn.ModuleList([
            build_block(embed_dim, compress_dim) for _ in range(num_scales)
        ])
        self.projection = nn.Sequential(
            nn.Conv2d(compress_dim * num_scales, embed_dim, kernel_size=1, bias=False),
            nn.GroupNorm(32, embed_dim)
        )
        self.learnable_gamma = nn.Parameter(torch.zeros(1))
        self._init_weights()

    def _init_weights(self):
        for module in self.compress_layers:
            nn.init.kaiming_normal_(module[0].weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(module[1].weight, 1.0)
            nn.init.constant_(module[1].bias, 0.0)
        nn.init.kaiming_normal_(self.projection[0].weight, mode='fan_out')

        nn.init.constant_(module[1].weight, 0.0)
        nn.init.constant_(module[1].bias, 0.0)

    def forward(self, f3, f6, f9, f12, shape):
        B, L, C = f12.shape
        H, W = shape
        f3_2d = f3[:, 1:].permute(0, 2, 1).reshape(B, C, H, W)
        f6_2d = f6[:, 1:].permute(0, 2, 1).reshape(B, C, H, W)
        f9_2d = f9[:, 1:].permute(0, 2, 1).reshape(B, C, H, W)
        f12_2d = f12[:, 1:].permute(0, 2, 1).reshape(B, C, H, W)

        f3_c = self.compress_layers[0](f3_2d)
        f6_c = self.compress_layers[1](f6_2d)
        f9_c = self.compress_layers[2](f9_2d)

        multi_scale = torch.cat([f3_c, f6_c, f9_c], dim=1)
        multi_scale = self.projection(multi_scale)
        scale = torch.tanh(self.learnable_gamma).abs() * self.max_scale
        fused_feat = f12_2d + scale * multi_scale
        fused_feat = F.normalize(fused_feat, p=2, dim=1)
        return fused_feat


# ==================== 频域感知空间偏差模块 (FASB) ====================
class FASBModule(nn.Module):
    """
    频域感知空间偏差模块 (Frequency-Aware Spatial Bias)
    替代原有的 PE 投影网络，利用遥感图像的频域特性建模内容依赖偏差
    """
    def __init__(self, in_channels, cnum, low_freq_threshold=0.2):
        super().__init__()
        self.in_channels = in_channels
        self.cnum = cnum
        self.threshold = low_freq_threshold
        # 低频分支：建模全局语义分布
        self.low_freq_branch = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=1)
        )
        # 高频分支：建模边界、细节、纹理
        self.high_freq_branch = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=1)
        )
        # 融合投影层生成 Mp
        self.fusion_proj = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, cnum, kernel_size=1)
        )
        # 可学习融合因子
        self.fusion_alpha = nn.Parameter(torch.ones(1) * 0.5)
        self._init_weights()
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def forward(self, x):
        """
        Args:
            x: 内容特征 [B, C, H, W] (ViT 最后一层 patch features)
        Returns:
            Mp: 内容依赖空间偏差 [B, cnum, H, W]
        """
        B, C, H, W = x.shape
        # 2D 快速傅里叶变换
        x_fft = torch.fft.fft2(x, norm='ortho')
        x_fft_shifted = torch.fft.fftshift(x_fft)

        # 创建频域掩码
        mask = torch.zeros((B, C, H, W), device=x.device)
        center_h, center_w = H // 2, W // 2
        radius = max(int(min(H, W) * self.threshold), 1)

        # 低频区域掩码 (中心圆)
        y = torch.arange(H, dtype=torch.float32, device=x.device).view(-1, 1)
        x_idx = torch.arange(W, dtype=torch.float32, device=x.device).view(1, -1)
        dist_from_center = torch.sqrt((y - center_h) ** 2 + (x_idx - center_w) ** 2)
        mask[:, :, dist_from_center <= radius] = 1.0

        # 频率分解
        low_fft = x_fft_shifted * mask
        high_fft = x_fft_shifted * (1.0 - mask)

        # 逆变换回到空间域
        low_feat = torch.fft.ifft2(torch.fft.ifftshift(low_fft), norm='ortho').real
        high_feat = torch.fft.ifft2(torch.fft.ifftshift(high_fft), norm='ortho').real

        # 分支提取
        f_l = self.low_freq_branch(low_feat)
        f_h = self.high_freq_branch(high_feat)

        # 融合生成最终偏差 Mp
        alpha = torch.sigmoid(self.fusion_alpha)
        combined = torch.cat([f_l * alpha, f_h * (1 - alpha)], dim=1)
        mp = self.fusion_proj(combined)
        return mp

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", nn.GELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=True, attn_mask=None)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attn(self.ln_1(x), self.ln_1(x), self.ln_1(x), need_weights=True, attn_mask=None)[0]
        x = x + self.mlp(self.ln_2(x))
        return x

    def _initialize_weights(self, clip_model, i):
        self.ln_1 = clip_model.visual.transformer.resblocks[i].ln_1
        self.ln_1.eps = 1e-06
        self.attn = clip_model.visual.transformer.resblocks[i].attn.to(torch.float32)
        self.attn.batch_first = True
        self.mlp = clip_model.visual.transformer.resblocks[i].mlp.to(torch.float32)
        self.ln_2 = clip_model.visual.transformer.resblocks[i].ln_2
        self.ln_2.eps = 1e-06
        for p in self.parameters():
            p.requires_grad = False
        return


class LastResidualAttentionBlock(nn.Module):
    def __init__(self, clip_model: clip, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", nn.GELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = nn.LayerNorm(d_model)
        self.attn_mask = attn_mask

        self._initialize_weights(clip_model)

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        y = self.ln_1(x)
        y = F.linear(y, self.attn.in_proj_weight, self.attn.in_proj_bias)
        N, L, C = y.shape
        y = y.view(N, L, 3, C // 3).permute(2, 0, 1, 3).reshape(3 * N, L, C // 3)
        y = F.linear(y, self.attn.out_proj.weight, self.attn.out_proj.bias)
        q, k, v = y.tensor_split(3, dim=0)
        v += x
        v = v + self.mlp(self.ln_2(v))
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        out = [x, q, k, v]
        return out

    def _initialize_weights(self, clip_model):
        self.ln_1 = clip_model.visual.transformer.resblocks[11].ln_1
        self.ln_1.eps = 1e-06
        self.attn = clip_model.visual.transformer.resblocks[11].attn.to(torch.float32)
        self.attn.batch_first = True
        self.mlp = clip_model.visual.transformer.resblocks[11].mlp.to(torch.float32)
        # self.mlp[1] = nn.GELU()
        self.ln_2 = clip_model.visual.transformer.resblocks[11].ln_2
        self.ln_2.eps = 1e-06
        for p in self.parameters():
            p.requires_grad = False
        return

class Transformer(nn.Module):
    def __init__(self, clip_model: clip, width: int, layers: int, heads: int,
                 attn_mask: torch.Tensor = None, extract_multiscale=False):
        super().__init__()
        self.width = width
        self.layers = layers
        self.extract_multiscale = extract_multiscale
        self.resblock = []
        for i in range(self.layers - 1):
            self.resblock.append(ResidualAttentionBlock(width, heads, attn_mask))
        self.resblock.append(LastResidualAttentionBlock(clip_model, width, heads, attn_mask))
        self._initialize_weights(clip_model)
        self.resblocks = nn.Sequential(*self.resblock)

    def forward(self, x: torch.Tensor):
        """
        Returns:
            如果extract_multiscale=True: (z, q, k, v, [f3, f6, f9, f12])
            否则: (z, q, k, v)
        """
        intermediate_feats = []

        for i, block in enumerate(self.resblocks):
            if i == len(self.resblocks) - 1:
                # 最后一层 (Block 12)
                z, q, k, v = block(x)
                if self.extract_multiscale:
                    # ！！！提取 Block 12 的输出 v ！！！
                    intermediate_feats.append(v.clone())
            else:
                x = block(x)
                # 记录第 3, 6, 9 层的输出 (索引 2, 5, 8)
                if self.extract_multiscale and i in [2, 5, 8]:
                    intermediate_feats.append(x.clone())

        if self.extract_multiscale:
            return z, q, k, v, intermediate_feats
        return z, q, k, v

    def _initialize_weights(self, clip_model):
        for i in range(self.layers - 1):
            self.resblock[i]._initialize_weights(clip_model, i)
        return


class VisionTransformer(nn.Module):
    def __init__(self, clip_model: clip, input_resolution: int, patch_size: int,
                 width: int, layers: int, heads: int, output_dim: int,
                 use_multiscale=False, compress_dim=256, max_scale=0.5):
        super().__init__()
        self.patch_size = patch_size
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.dilation = [1, 1]
        self.use_multiscale = use_multiscale

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width,
                               kernel_size=self.patch_size, stride=self.patch_size, bias=False)

        self.cls_token = torch.load('utils/cls_token.pt')
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(
            scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = nn.LayerNorm(width)
        self.ln_post = nn.LayerNorm(width)
        self.proj = clip_model.visual.proj.to(torch.float32)

        # 使用支持多尺度的Transformer
        self.transformer = Transformer(clip_model, width, layers, heads,
                                       extract_multiscale=use_multiscale)

        # 多尺度融合模块
        if use_multiscale:
            self.multiscale_fusion = MultiScaleFusion(embed_dim=width, compress_dim=256, max_scale=max_scale)

        self._initialize_weights(clip_model)

    def forward(self, x, train=False, img_metas=None):
        B = x.shape[0]

        # PatchEmbed + Padding（保持原样）
        input_h, input_w = x.size()[-2:]
        kernel_h, kernel_w = (self.patch_size, self.patch_size)
        stride_h, stride_w = (self.patch_size, self.patch_size)
        output_h = math.ceil(input_h / stride_h)
        output_w = math.ceil(input_w / stride_w)
        pad_h = max((output_h - 1) * stride_h + kernel_h - input_h, 0)
        pad_w = max((output_w - 1) * stride_w + kernel_w - input_w, 0)

        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [0, pad_w, 0, pad_h])

        x = x.to(device)
        x = self.conv1(x)
        x = x.flatten(2).transpose(1, 2)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Positional Embedding
        positional_embedding = self.positional_embedding.unsqueeze(dim=0)
        pos_h = self.input_resolution // self.patch_size
        pos_w = self.input_resolution // self.patch_size
        cls_token_weight = positional_embedding[:, 0]
        pos_embed_weight = positional_embedding[:, (-1 * pos_h * pos_w):]
        pos_embed_weight = pos_embed_weight.reshape(
            1, pos_h, pos_w, positional_embedding.shape[2]).permute(0, 3, 1, 2)
        pos_embed_weight = F.interpolate(pos_embed_weight, size=(output_h, output_w),
                                         mode='bicubic', align_corners=False)
        cls_token_weight = cls_token_weight.unsqueeze(1)
        pos_embed_weight = torch.flatten(pos_embed_weight, 2).transpose(1, 2)
        positional_embedding = torch.cat((cls_token_weight, pos_embed_weight), dim=1)

        x = x + positional_embedding
        x = self.ln_pre(x)

        # Transformer with multiscale extraction
        if self.use_multiscale:
            z, q, k, v, intermediate_feats = self.transformer(x)

            # 多尺度融合
            f3 = self.ln_post(intermediate_feats[0])
            f6 = self.ln_post(intermediate_feats[1])
            f9 = self.ln_post(intermediate_feats[2])
            f12 = self.ln_post(intermediate_feats[3])

            v_fused = self.multiscale_fusion(f3, f6, f9, f12, shape=(output_h, output_w))

        else:
            z, q, k, v = self.transformer(x)
            v = self.ln_post(v)  # 原始分支也必须过 LN
            v_fused = v[:, 1:].permute(0, 2, 1).reshape(B, -1, output_h, output_w)

        # 后续处理（保持原样）
        z = self.ln_post(z)
        cls_token = z[:, 0]
        z_global = cls_token @ self.proj if self.proj is not None else None

        # 返回格式与原始一致
        return [v_fused, (output_h, output_w), z_global, k,
                positional_embedding[:, 1:, :]]

    def _initialize_weights(self, clip_model):
        self.conv1 = clip_model.visual.conv1.to(torch.float32)
        self.class_embedding = clip_model.visual.class_embedding
        self.positional_embedding = clip_model.visual.positional_embedding
        self.ln_pre = clip_model.visual.ln_pre
        self.ln_post = clip_model.visual.ln_post

        for p in self.parameters():
            p.requires_grad = False

        # 多尺度模块设为可学习
        if self.use_multiscale:
            for p in self.multiscale_fusion.parameters():
                p.requires_grad = True
        return


class TextEncoder(nn.Module):
    def __init__(self, clip_model, training=False, cfg=None, device=None):
        super().__init__()
        self.transformer = clip_model.transformer.to(torch.float32)
        self.token_embedding = clip_model.token_embedding.to(torch.float32)
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection.to(torch.float32)
        self.dtype = torch.float32
        self.device = device
        token = torch.zeros((1, 73), dtype=torch.int).to(self.device)
        prompt_token = self.token_embedding(token)
        for p in self.parameters():
            p.requires_grad = False
        self.prompt_token = nn.Parameter(prompt_token)
        self.weight = False

        if not training:
            # prompt_token = torch.load(cfg.LOAD_PATH)['text_encoder.prompt_token']
            prompt_token = torch.load(cfg.LOAD_PATH)['module.text_encoder.prompt_token']
            self.prompt_token = nn.Parameter(prompt_token, requires_grad=False)

    def forward(self, cls_name_token):
        device = self.device
        prompt_token = self.prompt_token.repeat(cls_name_token.shape[0], 1, 1).to(device)
        cls_name_token = cls_name_token.to(device)

        start_token = self.token_embedding(torch.tensor(49406, dtype=torch.int, device=device)).repeat(
            cls_name_token.shape[0], 1, 1).to(device)
        cls_token = self.token_embedding(cls_name_token).to(device)
        x = torch.cat([start_token, prompt_token, cls_token], dim=1)
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), 74 + cls_name_token.argmax(dim=-1)] @ self.text_projection
        return x

# ==================== 第三步： ReCLIP 类 ====================
class ReCLIP(nn.Module):
    def __init__(self, cfg, clip_model, rank, zeroshot_weights,
                 use_multiscale=False, compress_dim=256, max_scale=0.5):
        super(ReCLIP, self).__init__()
        self.cnum = cfg.DATASET.NUM_CLASSES
        self.device = rank
        self.use_fasb = cfg.MODEL.get('USE_FASB', False)
        self.clip = clip_model
        self.k = cfg.DATASET.K
        self.logit_scale = clip_model.logit_scale

        self.vit = VisionTransformer(
            clip_model=clip_model,
            input_resolution=224,
            patch_size=16,
            width=768,
            layers=12,
            heads=12,
            output_dim=768,
            use_multiscale=use_multiscale,
            compress_dim=compress_dim,
            max_scale=max_scale
        )

        visual_channel = cfg.MODEL.VISUAL_CHANNEL
        text_channel = cfg.MODEL.TEXT_CHANNEL
        self.proj = nn.Conv2d(visual_channel, text_channel, 1, bias=False)
        self._initialize_weights(clip_model)

        if self.use_fasb:
            fasb_threshold = cfg.MODEL.get('FASB_LOW_FREQ_THRESHOLD', 0.2)
            self.fasb_module = FASBModule(in_channels=768,
                                          cnum=self.cnum,
                                          low_freq_threshold=fasb_threshold).to(self.device)
        for p in self.parameters():
            p.requires_grad = False
        # 多尺度模块
        if use_multiscale:
            for p in self.vit.multiscale_fusion.parameters():
                p.requires_grad = True
        # FASB模块
        if self.use_fasb:
            for p in self.fasb_module.parameters():
                p.requires_grad = True

        self.text_encoder = TextEncoder(clip_model, training=cfg.MODEL.TRAINING,
                                        cfg=cfg, device=rank)

    def forward(self, image, gt_cls, zeroshot_weights, cls_name_token,
                training=False, img_metas=None, return_feat=False):
        cnum = zeroshot_weights.shape[0]
        device = self.device
        gt_cls_text_embeddings = zeroshot_weights.to(device)

        batch_size = image.shape[0]
        image = image.to(device)

        # 1. 提取视觉特征 (v 为多尺度融合后的特征)
        # v: [B, 768, H, W], shape: (H, W), positional_embedding不再用于偏差建模
        v_fused, shape, z_global, k, _ = self.vit(image, train=False, img_metas=img_metas)

        feat = self.proj(v_fused)  # [B, 512, H, W]
        feat = feat / feat.norm(dim=1, keepdim=True)

        logit_scale = self.logit_scale.exp()

        # 2. 计算文本相似度 output_q
        output_q = F.conv2d(feat, gt_cls_text_embeddings[:, :, None, None]).permute(0, 2, 3, 1).reshape(batch_size, -1,
                                                                                                        cnum)

        # 3. 计算参考提示相似度 output_r
        prompt = self.text_encoder(cls_name_token)
        prompt = prompt / prompt.norm()
        output_r = F.conv2d(feat, prompt[:, :, None, None]).permute(0, 2, 3, 1).reshape(batch_size, -1, cnum)


        if self.use_fasb:
            # 使用融合后的视觉内容v生成 Mp (代替原来的 pe)
            mp = self.fasb_module(v_fused).permute(0, 2, 3, 1).reshape(batch_size, -1, cnum)
            output_r = torch.add(output_r, mp)

        output = torch.sub(output_q, output_r).permute(0, 2, 1).reshape(batch_size, -1, shape[0], shape[1])

        if return_feat:
            return output[0], feat[0], shape

        if training:
            output_scale = torch.mul(output.reshape(batch_size, cnum, -1).permute(0, 2, 1), 1000)
            output_gumbel = F.gumbel_softmax(output_scale, tau=1, hard=True, dim=2).reshape(batch_size, shape[0],
                                                                                            shape[1], -1)

            loss = 0
            for j in range(batch_size):
                masked_image_features = []
                if len(gt_cls[j]) == 0:
                    continue
                for i in gt_cls[j]:
                    mask = output_gumbel[j, :, :, i].unsqueeze(dim=0)
                    masked_image_feature = torch.mul(feat[j].unsqueeze(dim=0), mask)
                    feature_pool = nn.AdaptiveAvgPool2d((1, 1))(masked_image_feature).reshape(1, 512)
                    masked_image_features.append(feature_pool)
                masked_image_features = torch.stack(masked_image_features, dim=0).squeeze(dim=1)

                similarity_img = logit_scale * masked_image_features @ gt_cls_text_embeddings.t()
                labels = torch.tensor(gt_cls[j]).to(device)
                loss += F.cross_entropy(similarity_img, labels)
            return output, loss / batch_size

        return output

    def _initialize_weights(self, clip_model):
        self.proj.weight = nn.Parameter(
            clip_model.visual.proj[:, :, None, None].permute(1, 0, 2, 3).to(torch.float32),
            requires_grad=False)