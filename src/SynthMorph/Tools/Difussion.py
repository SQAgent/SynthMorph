# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 21:30:25 2025

@author: Administrator
"""
import time
import pickle
import os
import numpy as np
# import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from torch.nn.functional import normalize
import matplotlib.pyplot as plt
from PIL import Image
# import random
import torch.nn.functional as F
import math
import json
from tqdm import tqdm
import warnings
import cv2
import sys
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

class TopologyDataset(Dataset):
    def __init__(self, data_folder, sheet_names, C_sheet_names, 
                 model=None, device=None, 
                 embeddings_cache_path="matrix_embeddings_cache.pkl", 
                 force_recompute=False):
        
        self.data_folder = data_folder
        self.sheet_names = sheet_names
        self.C_sheet_names = C_sheet_names
        self.file_paths = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith('.xlsx')]
        self.num_picture = len(self.file_paths) * len(self.sheet_names)
        self.model = model
        self.device = device
        self.embeddings_cache_path = embeddings_cache_path
        
        # 加载或计算矩阵嵌入
        self.precomputed_embeddings = self._load_or_compute_embeddings(force_recompute)
        
    def _load_or_compute_embeddings(self, force_recompute=False):
        """加载或计算矩阵嵌入向量"""
        # 检查缓存文件是否存在且不需要重新计算
        if os.path.exists(self.embeddings_cache_path) and not force_recompute:
            print(f"从缓存文件 {self.embeddings_cache_path} 加载矩阵嵌入...")
            start_time = time.time()
            with open(self.embeddings_cache_path, 'rb') as f:
                cache_data = pickle.load(f)
                
            # 验证缓存数据是否与当前数据集匹配
            if (cache_data['num_files'] == len(self.file_paths) and 
                cache_data['num_sheets'] == len(self.sheet_names)):
                print(f"加载完成，耗时 {time.time() - start_time:.2f} 秒")
                return cache_data['embeddings']
            else:
                print("缓存数据与当前数据集不匹配，重新计算嵌入...")
        
        # 计算所有矩阵嵌入
        print("计算矩阵嵌入...")
        start_time = time.time()
        embeddings = []
        
        for file_idx, file_path in enumerate(self.file_paths):
            print(f"处理文件 {file_idx+1}/{len(self.file_paths)}: {os.path.basename(file_path)}")
            
            for sheet_idx, C_sheet_name in enumerate(self.C_sheet_names):
                # 读取弹性刚度矩阵C
                C_data = pd.read_excel(file_path, sheet_name=C_sheet_name, header=None)
                matrices = torch.tensor(C_data.values, dtype=torch.float32).unsqueeze(0)
                
                if self.model is not None and self.device is not None:
                    matrices = matrices.to(self.device)
                    with torch.no_grad():
                        matrix_embeddings = self.model.matrix_encoder(matrices)
                        matrix_embeddings = normalize(matrix_embeddings, dim=1)
                    embeddings.append(matrix_embeddings.cpu().numpy()[0])
                else:
                    # 如果没有模型，返回原始矩阵
                    embeddings.append(matrices.cpu().numpy()[0])
        
        embeddings = np.array(embeddings)
        
        # 保存到缓存文件
        cache_data = {
            'embeddings': embeddings,
            'num_files': len(self.file_paths),
            'num_sheets': len(self.sheet_names),
            'file_paths': self.file_paths,
            'sheet_names': self.sheet_names,
            'C_sheet_names': self.C_sheet_names
        }
        
        with open(self.embeddings_cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        
        print(f"嵌入计算完成并已保存到缓存，耗时 {time.time() - start_time:.2f} 秒")
        return embeddings
    
    def __len__(self):
        return self.num_picture

    def __getitem__(self, idx):
        file_idx = idx // len(self.sheet_names)
        sheet_idx = idx % len(self.sheet_names)
        
        file_path = self.file_paths[file_idx]
        sheet_name = self.sheet_names[sheet_idx]
        
        # 读取密度场数据并转换为图像
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
        data = 1.0 - df.values.astype(np.float32)
        
        image = Image.fromarray((data * 255).astype(np.uint8))
        image = transforms.ToTensor()(image)
        
        # 获取预计算的矩阵嵌入
        matrix_embedding = torch.tensor(self.precomputed_embeddings[idx], dtype=torch.float32)
        
        return image, matrix_embedding


# Image Enconder
class ImageEncoder(nn.Module):
    def __init__(self, embedding_dim=128):
        super(ImageEncoder, self).__init__()
        # 使用新的API加载预训练的ResNet18
        self.cnn = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # 修改第一层卷积以接受单通道输入
        self.cnn.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # 替换最后的全连接层
        self.cnn.fc = nn.Sequential(
            nn.Linear(self.cnn.fc.in_features, 512),
            nn.ReLU(),
            nn.Linear(512, embedding_dim))

    def forward(self, x):
        return self.cnn(x)


# matrix Enconder
class MatrixEncoder(nn.Module):
    def __init__(self, input_dim=9, embedding_dim=128):
        super(MatrixEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim)
        )

    def forward(self, x):
        # Flatten to 9D
        x = x.reshape(x.size(0), -1)
        return self.encoder(x)


# 多模态对比学习模型
class MultiModalContrastiveModel(nn.Module):
    def __init__(self, image_embedding_dim=128, matrix_embedding_dim=128):
        super(MultiModalContrastiveModel, self).__init__()
        self.image_encoder = ImageEncoder(embedding_dim=image_embedding_dim)
        self.matrix_encoder = MatrixEncoder(embedding_dim=matrix_embedding_dim)

    def forward(self, images, matrices):
        image_embeddings = self.image_encoder(images)
        matrix_embeddings = self.matrix_encoder(matrices)
        return image_embeddings, matrix_embeddings


# MLP
class MLPRegressor(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLPRegressor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 70),
            nn.ReLU(),
            nn.Linear(70, 40),
            nn.ReLU(),
            nn.Linear(40, 20),
            nn.ReLU(),
            nn.Linear(20, output_size)
        )

    def forward(self, x):
        return self.network(x)


class AdaLN(nn.Module):
    def __init__(self, num_features, cond_dim):
        super(AdaLN, self).__init__()
        self.num_features = num_features
        self.norm = nn.GroupNorm(8, num_features, affine=False)
        self.linear = nn.Linear(cond_dim, num_features * 2)

    def forward(self, x, cond):
        x = self.norm(x)
        cond_out = self.linear(cond)
        scale, bias = cond_out.chunk(2, dim=1)
        scale = scale.unsqueeze(-1).unsqueeze(-1)
        bias = bias.unsqueeze(-1).unsqueeze(-1)
        x = x * (1 + scale) + bias
        return x


class ConditionalUNetWithAdaLN(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, cond_dim=128, base_channels=64):
        super(ConditionalUNetWithAdaLN, self).__init__()

        self.base_channels = base_channels
        self.cond_dim = cond_dim

        # 时间步嵌入
        self.time_embed = nn.Sequential(
            nn.Linear(base_channels, base_channels * 2),
            nn.SiLU(),
            nn.Linear(base_channels * 2, base_channels * 2))

        # 条件投影网络
        self.cond_proj_enc1 = nn.Linear(cond_dim, base_channels * 2)
        self.cond_proj_enc2 = nn.Linear(cond_dim, base_channels * 2)
        self.cond_proj_enc3 = nn.Linear(cond_dim, base_channels * 2)
        self.cond_proj_enc4 = nn.Linear(cond_dim, base_channels * 2)
        self.cond_proj_bottleneck = nn.Linear(cond_dim, base_channels * 2)
        self.cond_proj_dec4 = nn.Linear(cond_dim, base_channels * 2)
        self.cond_proj_dec3 = nn.Linear(cond_dim, base_channels * 2)
        self.cond_proj_dec2 = nn.Linear(cond_dim, base_channels * 2)
        self.cond_proj_dec1 = nn.Linear(cond_dim, base_channels * 2)

        # 编码器
        self.enc1_conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=3,
                                    padding=1, padding_mode='circular')
        self.enc1_adaln1 = AdaLN(base_channels, base_channels * 4)
        self.enc1_conv2 = nn.Conv2d(base_channels, base_channels, kernel_size=3,
                                    padding=1, padding_mode='circular')
        self.enc1_adaln2 = AdaLN(base_channels, base_channels * 4)

        self.enc2_conv1 = nn.Conv2d(base_channels, base_channels * 2,
                                    kernel_size=3, padding=1,
                                    padding_mode='circular')
        self.enc2_adaln1 = AdaLN(base_channels * 2, base_channels * 4)
        self.enc2_conv2 = nn.Conv2d(base_channels * 2, base_channels * 2,
                                    kernel_size=3, padding=1,
                                    padding_mode='circular')
        self.enc2_adaln2 = AdaLN(base_channels * 2, base_channels * 4)

        self.enc3_conv1 = nn.Conv2d(base_channels * 2, base_channels * 4,
                                    kernel_size=3, padding=1,
                                    padding_mode='circular')
        self.enc3_adaln1 = AdaLN(base_channels * 4, base_channels * 4)
        self.enc3_conv2 = nn.Conv2d(base_channels * 4, base_channels * 4,
                                    kernel_size=3, padding=1,
                                    padding_mode='circular')
        self.enc3_adaln2 = AdaLN(base_channels * 4, base_channels * 4)

        self.enc4_conv1 = nn.Conv2d(base_channels * 4, base_channels * 8,
                                    kernel_size=3, padding=1,
                                    padding_mode='circular')
        self.enc4_adaln1 = AdaLN(base_channels * 8, base_channels * 4)
        self.enc4_conv2 = nn.Conv2d(base_channels * 8, base_channels * 8,
                                    kernel_size=3, padding=1,
                                    padding_mode='circular')
        self.enc4_adaln2 = AdaLN(base_channels * 8, base_channels * 4)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # 瓶颈层
        self.bottleneck_conv1 = nn.Conv2d(base_channels * 8, base_channels * 16,
                                          kernel_size=3, padding=1,
                                          padding_mode='circular')
        self.bottleneck_adaln1 = AdaLN(base_channels * 16, base_channels * 4)
        self.bottleneck_conv2 = nn.Conv2d(base_channels * 16, base_channels * 16,
                                          kernel_size=3, padding=1,
                                          padding_mode='circular')
        self.bottleneck_adaln2 = AdaLN(base_channels * 16, base_channels * 4)

        # 解码器
        self.up4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8,
                                      kernel_size=2, stride=2)
        self.dec4_conv1 = nn.Conv2d(base_channels * 16, base_channels * 8,
                                    kernel_size=3, padding=1,
                                    padding_mode='circular')
        self.dec4_adaln1 = AdaLN(base_channels * 8, base_channels * 4)
        self.dec4_conv2 = nn.Conv2d(base_channels * 8, base_channels * 8,
                                    kernel_size=3, padding=1,
                                    padding_mode='circular')
        self.dec4_adaln2 = AdaLN(base_channels * 8, base_channels * 4)

        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4,
                                      kernel_size=2, stride=2)
        self.dec3_conv1 = nn.Conv2d(base_channels * 8, base_channels * 4,
                                    kernel_size=3, padding=1,
                                    padding_mode='circular')
        self.dec3_adaln1 = AdaLN(base_channels * 4, base_channels * 4)
        self.dec3_conv2 = nn.Conv2d(base_channels * 4, base_channels * 4,
                                    kernel_size=3, padding=1,
                                    padding_mode='circular')
        self.dec3_adaln2 = AdaLN(base_channels * 4, base_channels * 4)

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2,
                                      kernel_size=2, stride=2)
        self.dec2_conv1 = nn.Conv2d(base_channels * 4, base_channels * 2,
                                    kernel_size=3, padding=1,
                                    padding_mode='circular')
        self.dec2_adaln1 = AdaLN(base_channels * 2, base_channels * 4)
        self.dec2_conv2 = nn.Conv2d(base_channels * 2, base_channels * 2,
                                    kernel_size=3, padding=1,
                                    padding_mode='circular')
        self.dec2_adaln2 = AdaLN(base_channels * 2, base_channels * 4)

        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels,
                                      kernel_size=2, stride=2)
        self.dec1_conv1 = nn.Conv2d(base_channels * 2, base_channels,
                                    kernel_size=3, padding=1,
                                    padding_mode='circular')
        self.dec1_adaln1 = AdaLN(base_channels, base_channels * 4)
        self.dec1_conv2 = nn.Conv2d(base_channels, base_channels,
                                    kernel_size=3, padding=1,
                                    padding_mode='circular')
        self.dec1_adaln2 = AdaLN(base_channels, base_channels * 4)

        self.final_conv = nn.ConvTranspose2d(base_channels, out_channels,
                                             kernel_size=4, stride=2,
                                             padding=1,
                                             output_padding=0)
        self.silu = nn.SiLU()

    def _get_timestep_embedding(self, timesteps, dim):
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if dim % 2 == 1:
            emb = F.pad(emb, (0, 1, 0, 0))
        return emb

    def _encoder_block(self, x, cond_embed, conv1, adaln1, conv2, adaln2, pool=True):
        x = conv1(x)
        x = adaln1(x, cond_embed)
        x = self.silu(x)
        x = conv2(x)
        x = adaln2(x, cond_embed)
        x = self.silu(x)
        if pool:
            x_pool = self.pool(x)
            return x, x_pool
        else:
            return x, x

    def _decoder_block(self, x, cond_embed, conv1, adaln1, conv2, adaln2):
        x = conv1(x)
        x = adaln1(x, cond_embed)
        x = self.silu(x)
        x = conv2(x)
        x = adaln2(x, cond_embed)
        x = self.silu(x)
        return x

    def forward(self, x, t, cond):
        # 时间步嵌入
        t_embed = self._get_timestep_embedding(t, self.base_channels)
        t_embed = self.time_embed(t_embed)

        # 为每个层级准备条件嵌入
        cond_embed_enc1 = torch.cat(
            [self.cond_proj_enc1(cond), t_embed], dim=1)
        cond_embed_enc2 = torch.cat(
            [self.cond_proj_enc2(cond), t_embed], dim=1)
        cond_embed_enc3 = torch.cat(
            [self.cond_proj_enc3(cond), t_embed], dim=1)
        cond_embed_enc4 = torch.cat(
            [self.cond_proj_enc4(cond), t_embed], dim=1)
        cond_embed_bottleneck = torch.cat(
            [self.cond_proj_bottleneck(cond), t_embed], dim=1)
        cond_embed_dec4 = torch.cat(
            [self.cond_proj_dec4(cond), t_embed], dim=1)
        cond_embed_dec3 = torch.cat(
            [self.cond_proj_dec3(cond), t_embed], dim=1)
        cond_embed_dec2 = torch.cat(
            [self.cond_proj_dec2(cond), t_embed], dim=1)
        cond_embed_dec1 = torch.cat(
            [self.cond_proj_dec1(cond), t_embed], dim=1)

        # 编码器路径
        e1, e1_pool = self._encoder_block(x, cond_embed_enc1,
                                          self.enc1_conv1, self.enc1_adaln1,
                                          self.enc1_conv2, self.enc1_adaln2)
        e2, e2_pool = self._encoder_block(e1_pool, cond_embed_enc2,
                                          self.enc2_conv1, self.enc2_adaln1,
                                          self.enc2_conv2, self.enc2_adaln2)
        e3, e3_pool = self._encoder_block(e2_pool, cond_embed_enc3,
                                          self.enc3_conv1, self.enc3_adaln1,
                                          self.enc3_conv2, self.enc3_adaln2)
        e4, e4_pool = self._encoder_block(e3_pool, cond_embed_enc4,
                                          self.enc4_conv1, self.enc4_adaln1,
                                          self.enc4_conv2, self.enc4_adaln2)

        # 瓶颈层
        bottleneck_input = self.pool(e4_pool)
        b = self.bottleneck_conv1(bottleneck_input)
        b = self.bottleneck_adaln1(b, cond_embed_bottleneck)
        b = self.silu(b)
        b = self.bottleneck_conv2(b)
        b = self.bottleneck_adaln2(b, cond_embed_bottleneck)
        b = self.silu(b)

        # 解码器路径
        d4 = self.up4(b)
        d4 = torch.cat([d4, e4_pool], dim=1)
        d4 = self._decoder_block(d4, cond_embed_dec4,
                                 self.dec4_conv1, self.dec4_adaln1,
                                 self.dec4_conv2, self.dec4_adaln2)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3_pool], dim=1)
        d3 = self._decoder_block(d3, cond_embed_dec3,
                                 self.dec3_conv1, self.dec3_adaln1,
                                 self.dec3_conv2, self.dec3_adaln2)

        d2 = self.up2(d3)
        if d2.shape[2:] != e2_pool.shape[2:]:
            d2 = F.interpolate(d2, size=e2_pool.shape[2:],
                               mode='bilinear', align_corners=False)
        d2 = torch.cat([d2, e2_pool], dim=1)
        d2 = self._decoder_block(d2, cond_embed_dec2,
                                 self.dec2_conv1, self.dec2_adaln1,
                                 self.dec2_conv2, self.dec2_adaln2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1_pool], dim=1)
        d1 = self._decoder_block(d1, cond_embed_dec1,
                                 self.dec1_conv1, self.dec1_adaln1,
                                 self.dec1_conv2, self.dec1_adaln2)

        output = self.final_conv(d1)
        return output


class DiffusionInference:
    def __init__(self, model_path, config_path, device="cuda", timesteps=1000):
        self.device = device
        self.timesteps = timesteps

        # 加载模型配置
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        print(f"加载模型配置: {self.config}")

        # 初始化模型
        self.model = ConditionalUNetWithAdaLN(
            in_channels=self.config['in_channels'],
            out_channels=self.config['out_channels'],
            cond_dim=self.config['cond_dim'],
            base_channels=self.config['base_channels']
        ).to(device)

        # 加载模型权重
        if model_path.endswith('.pth'):
            checkpoint = torch.load(model_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                # 完整检查点文件
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print(f"从检查点加载模型，训练轮数: {checkpoint.get('epoch', '未知')}")
            else:
                # 仅权重文件
                self.model.load_state_dict(checkpoint)
                print("加载模型权重文件")
        else:
            raise ValueError("模型文件格式不支持")

        self.model.eval()
        print("模型加载完成，进入评估模式")

        # 设置扩散参数（与训练时一致）
        self._setup_diffusion_parameters()

    def _setup_diffusion_parameters(self):
        """设置扩散过程参数（与训练代码保持一致）"""
        beta_start = 1e-4
        beta_end = 0.02

        self.betas = torch.linspace(beta_start, beta_end, self.timesteps)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(
            self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(
            1. - self.alphas_cumprod)

        # 计算后验方差
        self.posterior_variance = self.betas * \
            (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

        # 转移到设备
        self.betas = self.betas.to(self.device)
        self.alphas = self.alphas.to(self.device)
        self.alphas_cumprod = self.alphas_cumprod.to(self.device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(self.device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(self.device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(
            self.device)
        self.posterior_variance = self.posterior_variance.to(self.device)

    def _extract(self, a, t, x_shape):
        """从张量a中提取时间步t对应的值"""
        batch_size = t.shape[0]
        out = a.gather(-1, t.to(self.device)).to(t.device)  # 修复：确保在正确设备上
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    @torch.no_grad()
    def p_sample(self, x, t, cond):
        """从p(x_{t-1} | x_t)采样 - 修复版本"""
        # 预测噪声
        pred_noise = self.model(x, t, cond)

        # 计算均值
        sqrt_alpha_cumprod_t = self._extract(
            self.sqrt_alphas_cumprod, t, x.shape)
        sqrt_one_minus_alpha_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape)

        # 估计原始图像 x0
        pred_x0 = (x - sqrt_one_minus_alpha_cumprod_t *
                   pred_noise) / sqrt_alpha_cumprod_t
        pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)

        # 修复：使用提取的时间步参数而不是整个序列
        alpha_t = self._extract(self.alphas, t, x.shape)
        alpha_cumprod_t = self._extract(self.alphas_cumprod, t, x.shape)
        alpha_cumprod_prev_t = self._extract(
            self.alphas_cumprod_prev, t, x.shape)
        beta_t = self._extract(self.betas, t, x.shape)

        # 计算后验均值的系数 - 修复版本
        posterior_mean_coef1 = beta_t * \
            torch.sqrt(alpha_cumprod_prev_t) / (1. - alpha_cumprod_t)
        posterior_mean_coef2 = (1. - alpha_cumprod_prev_t) * \
            torch.sqrt(alpha_t) / (1. - alpha_cumprod_t)

        # 计算后验均值
        posterior_mean = posterior_mean_coef1 * pred_x0 + posterior_mean_coef2 * x

        if t[0] == 0:
            return posterior_mean
        else:
            # 添加噪声
            noise = torch.randn_like(x)
            posterior_variance_t = self._extract(
                self.posterior_variance, t, x.shape)
            return posterior_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def generate(self, cond_embedding, num_samples=10, image_size=100, save_path=None,
                 show_progress=True, save_intermediate=False, intermediate_steps=20,
                 intermediate_dir="generation_process"):
        """
        生成图像，可选择保存中间过程
        Args:
            cond_embedding: 条件嵌入向量 [cond_dim] 或 [batch_size, cond_dim]
            num_samples: 生成样本数量
            image_size: 图像尺寸
            save_path: 保存路径（可选）
            show_progress: 是否显示进度条
            save_intermediate: 是否保存中间过程图像
            intermediate_steps: 保存中间过程的步数（每隔多少步保存一次）
            intermediate_dir: 中间过程图像的保存目录
        Returns:
            生成的图像张量 [num_samples, 1, image_size, image_size] 和中间过程列表（如果保存）
        """
        # 处理条件嵌入
        if isinstance(cond_embedding, np.ndarray):
            cond_embedding = torch.tensor(
                cond_embedding, dtype=torch.float32, device=self.device)

        if cond_embedding.dim() == 1:
            cond_embedding = cond_embedding.unsqueeze(0).repeat(num_samples, 1)
        elif cond_embedding.dim() == 2 and cond_embedding.shape[0] == 1:
            cond_embedding = cond_embedding.repeat(num_samples, 1)

        assert cond_embedding.shape[0] == num_samples, "条件嵌入的batch_size必须与num_samples匹配"

        print(f"生成 {num_samples} 个样本，图像尺寸: {image_size}x{image_size}")
        print(f"条件嵌入形状: {cond_embedding.shape}")

        # 从随机噪声开始
        seed = 49
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            
        x = torch.randn(num_samples, 1, image_size,
                        image_size, device=self.device)

        # 存储中间过程
        intermediate_images = []

        # 反向扩散过程
        iterator = range(self.timesteps - 1, -1, -1)
        if show_progress:
            iterator = tqdm(iterator, desc="生成图像")

        for i in iterator:
            t = torch.full((num_samples,), i,
                           device=self.device, dtype=torch.long)
            x = self.p_sample(x, t, cond_embedding)

            # 保存中间过程
            if save_intermediate and (i % intermediate_steps == 0 or i == 0):
                # 将图像从[-1,1]转换到[0,1]
                intermediate_img = x  # (x + 1) / 2
                intermediate_img = torch.clamp(intermediate_img, 0, 1)
                intermediate_images.append(intermediate_img.cpu().clone())

        # 最终图像处理
        generated_images = x  # (x + 1) / 2
        generated_images = torch.clamp(generated_images, 0, 1)

        # 保存最终图像
        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(
                save_path) else '.', exist_ok=True)
            self._save_images(generated_images, save_path)

        # 保存中间过程图像
        if save_intermediate and intermediate_images:
            self._save_intermediate_images(
                intermediate_images, intermediate_dir)

        return generated_images, intermediate_images

    def _save_images(self, images, save_path):
        """保存生成的图像"""
        if images.shape[0] == 1:
            # 单张图像
            img = images[0, 0].cpu().numpy()
            img = (img * 255).astype(np.uint8)
            # 简单的线性映射
            start_color = np.array([83, 111, 122], dtype=np.float32)
            end_color = np.array([255, 255, 255], dtype=np.float32)
            
            # 将灰度值归一化到[0,1]
            gray_normalized = img / 255.0
            
            # 创建彩色图像
            height, width = img.shape
            colored_img = np.zeros((height, width, 3), dtype=np.uint8)
            
            # 对每个通道应用线性插值
            for c in range(3):
                channel_values = start_color[c] + gray_normalized * (end_color[c] - start_color[c])
                colored_img[..., c] = np.clip(channel_values, 0, 255).astype(np.uint8)
            
            Image.fromarray(img).save(save_path)
            print(f"图像已保存: {save_path}")
        else:
            # 多张图像
            base_name, ext = os.path.splitext(save_path)
            for i in range(images.shape[0]):
                img = images[i, 0].cpu().numpy()
                img = (img * 255).astype(np.uint8)
                # 简单的线性映射
                start_color = np.array([83, 111, 122], dtype=np.float32)
                end_color = np.array([255, 255, 255], dtype=np.float32)
                
                # 将灰度值归一化到[0,1]
                gray_normalized = img / 255.0
                
                # 创建彩色图像
                height, width = img.shape
                colored_img = np.zeros((height, width, 3), dtype=np.uint8)
                
                # 对每个通道应用线性插值
                for c in range(3):
                    channel_values = start_color[c] + gray_normalized * (end_color[c] - start_color[c])
                    colored_img[..., c] = np.clip(channel_values, 0, 255).astype(np.uint8)
                img_path = f"{base_name}_{i+1:03d}{ext}"
                
                Image.fromarray(img).save(img_path)
            print(
                f"{images.shape[0]} 张图像已保存到: {base_name}_*.{ext.split('.')[-1]}")

    def _save_intermediate_images(self, intermediate_images, save_dir):
        """保存中间过程图像"""
        os.makedirs(save_dir, exist_ok=True)

        for i, img_tensor in enumerate(intermediate_images):
            # 计算对应的时间步
            timestep = len(intermediate_images) - 1 - i
            actual_timestep = timestep * \
                (self.timesteps // len(intermediate_images))

            # 保存每张中间图像
            for j in range(img_tensor.shape[0]):
                img = img_tensor[j, 0].numpy()
                img = (img * 255).astype(np.uint8)
                # 简单的线性映射
                start_color = np.array([83, 111, 122], dtype=np.float32)
                end_color = np.array([255, 255, 255], dtype=np.float32)
                
                # 将灰度值归一化到[0,1]
                gray_normalized = img / 255.0
                
                # 创建彩色图像
                height, width = img.shape
                colored_img = np.zeros((height, width, 3), dtype=np.uint8)
                
                # 对每个通道应用线性插值
                for c in range(3):
                    channel_values = start_color[c] + gray_normalized * (end_color[c] - start_color[c])
                    colored_img[..., c] = np.clip(channel_values, 0, 255).astype(np.uint8)

                filename = f"step_{actual_timestep:04d}_sample_{j+1:03d}.png"
                filepath = os.path.join(save_dir, filename)
                Image.fromarray(colored_img).save(filepath)

        print(f"中间过程图像已保存到: {save_dir}")

        # 创建过程动画（可选）
        self._create_process_gif(intermediate_images, save_dir)

    def _create_process_gif(self, intermediate_images, save_dir):
        """创建生成过程的GIF动画"""
        try:
            images_for_gif = []
            for i, img_tensor in enumerate(intermediate_images):
                # 使用第一张样本创建GIF
                img = img_tensor[0, 0].numpy()
                img = (img * 255).astype(np.uint8)
                pil_img = Image.fromarray(img)
                images_for_gif.append(pil_img)

            # 保存为GIF
            gif_path = os.path.join(save_dir, "generation_process.gif")
            images_for_gif[0].save(
                gif_path,
                save_all=True,
                append_images=images_for_gif[1:],
                duration=150,  # 每帧500ms
                loop=0
            )
            print(f"生成过程GIF已保存: {gif_path}")
        except Exception as e:
            print(f"创建GIF失败: {e}")

    def generate_with_different_conditions(self, cond_embeddings, image_size=100, save_dir="generated_results"):
        """
        为不同的条件嵌入生成图像
        Args:
            cond_embeddings: 条件嵌入列表或数组 [num_conditions, cond_dim]
            image_size: 图像尺寸
            save_dir: 保存目录
        """
        os.makedirs(save_dir, exist_ok=True)

        if isinstance(cond_embeddings, list):
            cond_embeddings = torch.stack(cond_embeddings)

        results = []
        for i, cond_emb in enumerate(cond_embeddings):
            print(f"为条件 {i+1}/{len(cond_embeddings)} 生成图像...")

            # 为每个条件生成一个样本
            generated, _ = self.generate(
                cond_embedding=cond_emb.unsqueeze(0),
                num_samples=1,
                image_size=image_size,
                save_path=os.path.join(save_dir, f"condition_{i+1:03d}.png"),
                show_progress=False
            )

            results.append(generated)

        print(f"所有图像已生成并保存到: {save_dir}")
        return torch.cat(results, dim=0)


def Picture_Load(image_path):

    if image_path == None:
        image_path = input("请输入图片路径: ").strip()

    if not os.path.exists(image_path):
        print(f"错误: 文件 '{image_path}' 不存在")
        return None

    # 读取图片
    image = cv2.imread(image_path)

    if image is None:
        print(f"错误: 无法加载图片 '{image_path}'，请检查文件格式")
        return None

    # 获取图片尺寸
    height, width = image.shape[:2]
    print(f"图片尺寸: {width}×{height} 像素")

    # 检查图片尺寸是否为100x100
    if width != 100 or height != 100:

        print(f"错误: 图片尺寸不是100×100像素，而是{width}×{height}像素，将自动缩放...")
        image = cv2.resize(image, (100, 100), interpolation=cv2.INTER_AREA)
        print(f"✓ 已缩放为100×100像素 (直接缩放)")
        
    print("✓ 图片尺寸符合要求 (100×100像素)")

    # 转换为单通道灰度图
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        print("✓ 已转换为单通道灰度图")
    else:
        gray_image = image
        print("✓ 图片已经是单通道")

    # 检查灰度图的范围并转换为0-1范围的密度场矩阵
    # 根据输入图像的亮度范围进行归一化
    min_val = np.min(gray_image)
    max_val = np.max(gray_image)

    if max_val > min_val:
        # 归一化到0-1范围
        density_matrix = (gray_image.astype(np.float32) -
                          min_val) / (max_val - min_val)
        print(f"✓ 已归一化到0-1范围 (原始范围: {min_val}-{max_val})")
    else:
        # 如果所有像素值相同，创建全0或全1矩阵
        density_matrix = np.zeros((100, 100), dtype=np.float32)
        if min_val > 0:
            density_matrix[:, :] = 1.0
        print("✓ 创建常数值密度场矩阵")
    return density_matrix


def Predict_C_from_image(Picture_path, CL_path, MLP_path, device, output_dir="."):

    model_CL = MultiModalContrastiveModel(image_embedding_dim=128,
                                          matrix_embedding_dim=128).to(device)
    checkpoint = torch.load(CL_path,
                            map_location=device)
    model_CL.load_state_dict(checkpoint['model_state_dict'])

    print('model_CL_successful')

    image = Picture_Load(Picture_path)
    query_image = transforms.ToTensor()(image).unsqueeze(0)

    model_CL.eval()
    with torch.no_grad():
        query_embedding = model_CL.image_encoder(query_image.to(device))
        query_embedding = normalize(query_embedding, dim=1)

    print(f"Query image embedding : {query_embedding}")
    # print(query_embedding)

    checkpoint = torch.load(MLP_path,
                            map_location=device,
                            weights_only=False)
    model_MLP = MLPRegressor(checkpoint['input_size'],
                             checkpoint['output_size'])
    model_MLP.load_state_dict(checkpoint['model_state_dict'])
    model_MLP.to(device)
    model_MLP.eval()

    with torch.no_grad():
        predictions = model_MLP(query_embedding).cpu().numpy()[0].tolist()

    # Result print
    C11, C22, C33, C12, C13, C23 = predictions

    print(C11, '\t', C12 ,'\t', C13)
    print('\t'+'\t', '\t'+'\t', C22, '\t', C23)
    print('\t'+'\t', '\t'+'\t','\t'+'\t', '\t'+'\t', C33)
    C_matrix = [
            [C11, C12, C13],
            [C12, C22, C23],
            [C13, C23, C33],
        ]
    return  output_dir,C_matrix

def Generate_image_from_C(C, CL_path, Dif_path, config_path, device, output_dir="."):

    model_CL = MultiModalContrastiveModel(image_embedding_dim=128,
                                          matrix_embedding_dim=128).to(device)

    checkpoint = torch.load(CL_path,
                            map_location=device)
    model_CL.load_state_dict(checkpoint['model_state_dict'])

    # file_path = '/home/wmh/density_field/'+file_name+'.xlsx'

    if C is None:
        C = []
        for i in range(3):
            while True:
                try:
                    row_input = input(f"第{i+1}行: ")
                    row_values = [float(x) for x in row_input.split()]

                    if len(row_values) != 3:
                        print("每行需要输入3个数值，请重新输入")
                        continue

                    C.append(row_values)
                    break

                except ValueError:
                    print("输入格式错误，请输入三个用空格分隔的数值，如: 0.071 -0.032 -0.031")

        np.array(C)

    query_C = torch.tensor(C, dtype=torch.float32).unsqueeze(0).to(device)

    # print(query_C)
    model_CL.eval()
    with torch.no_grad():
        query_embedding = model_CL.matrix_encoder(query_C)
        query_embedding = normalize(query_embedding, dim=1)


    print(f"Query C embedding : {query_embedding}")


    # 初始化推理器
    model_path = Dif_path
    # config_path = pth_path + "model_config_adaln.json" #####

    try:
        diffusion_inference = DiffusionInference(
            model_path=model_path,
            config_path=config_path,
            device=device,
            timesteps=1000)
        save_path = os.path.join(output_dir, "generate.png")
        intermediate_dir = os.path.join(output_dir, "generation_process")
        generated_image, intermediate_images = diffusion_inference.generate(
            cond_embedding=query_embedding,
            num_samples=1,
            image_size=100,
            save_path=save_path,
            show_progress=True,
            save_intermediate=True,  # 启用中间过程保存
            intermediate_steps=50,   # 每50步保存一次
            intermediate_dir=intermediate_dir  # 中间过程保存目录
        )
        print(f"生成图像形状: {generated_image.shape}")
        print(f"保存了 {len(intermediate_images)} 张中间过程图像")
        print("\n=== 推理完成 ===")
        return generated_image, C

    except Exception as e:
        print(f"推理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


def get_file_and_sheet_info(dataset, indices):
    """获取指定索引对应的文件名和sheet名称"""
    result = []
    
    for idx in indices:
        if idx >= len(dataset):
            print(f"索引 {idx} 超出数据集范围（最大索引为 {len(dataset)-1}）")
            continue
            
        # 计算文件索引和sheet索引
        file_idx = idx // len(dataset.sheet_names)
        sheet_idx = idx % len(dataset.sheet_names)
        
        # 获取文件路径和sheet名称
        file_path = dataset.file_paths[file_idx]
        file_name = os.path.basename(file_path)  # 提取文件名
        sheet_name = dataset.sheet_names[sheet_idx]
        
        result.append({
            'index': idx,
            'file_name': file_name,
            'sheet_name': sheet_name,
            'file_path': file_path,
            'file_index': file_idx,
            'sheet_index': sheet_idx
        })
    
    return result


def load_split_info(filename="dataset_split_info.json"):
    """加载数据集划分信息并返回相关信息"""
    if not os.path.exists(filename):
        print(f"文件 {filename} 不存在")
        return None
    
    with open(filename, 'r') as f:
        split_info = json.load(f)
    
    print(f"从 {filename} 加载数据集划分信息")
    print(f"训练集大小: {split_info['train_size']} ({split_info['train_ratio']*100:.1f}%)")
    print(f"测试集大小: {split_info['test_size']} ({split_info['test_ratio']*100:.1f}%)")
    print(f"划分时间: {split_info['split_time']}")
    
    return split_info

def predict(Picture_path = None, C = None, output_dir = "."):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pth_path = '/home/shangqing/sqdata/model/wmh/'
    CL_model = 'enhanced_multimodal_contrastive_model_staged_2.pth'
    MLP_model = 'MLP_model_1.pth'
    Dif_model = "final_diffusion_model_adaln_weights_3.pth"
    config = "model_config_adaln_3.json"
    if Picture_path == None and C is None:
        print("请至少提供图片路径或弹性张量C")
        return
    if Picture_path is not None:
        output_path, C_end = Predict_C_from_image(Picture_path, pth_path+CL_model,
                              pth_path+MLP_model, device, output_dir)
    if C is not None:
        output_path, C_end =Generate_image_from_C(C, pth_path+CL_model, 
                              pth_path+Dif_model, 
                              pth_path+config, device, output_dir)
        
    return output_path, C_end
    
if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pth_path = '/home/shangqing/sqdata/model/wmh/'
    CL_model = 'enhanced_multimodal_contrastive_model_staged_2.pth'
    MLP_model = 'MLP_model_1.pth'
    Dif_model = "final_diffusion_model_adaln_weights_3.pth"
    config = "model_config_adaln_3.json"

    # # Function 1 ==================================
    # Picture_path = None
    # Picture_path = 'test.png'
    # Predict_C_from_image(Picture_path, pth_path+CL_model,
    #                       pth_path+MLP_model, device)
    

    
    # Function 2 ==================================


    C=np.array([[ 0.084476385,-0.037319064,-0.010734257],
                [-0.037319064,0.084589334,0.010550465],
                [-0.010734257,0.010550465,0.002932869]])
    
    Generate_image_from_C(C, pth_path+CL_model, 
                          pth_path+Dif_model, 
                          pth_path+config, device)
    

