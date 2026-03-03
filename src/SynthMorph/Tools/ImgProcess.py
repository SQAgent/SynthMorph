# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 13:44:26 2025

@author: Administrator
"""

import os
import cv2
import sys
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import xlwt


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

    # 检查图片尺寸是否为100x100，如果不是则自动缩放
    if width != 100 or height != 100:
        print(f"图片尺寸不是100×100像素，而是{width}×{height}像素，自动缩放到100×100像素")
        
        # 使用双线性插值缩放到100x100
        image = cv2.resize(image, (100, 100), interpolation=cv2.INTER_LINEAR)
        print("✓ 已自动缩放到100×100像素")
    else:
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

def from_image_to_contour(image_path, contour_path):
    
    
    density_data= Picture_Load(image_path)
    
    density_data = cv2.resize(
        density_data.astype(np.float32),  # 确保数据类型为float32
        (1000, 1000),                    # 目标尺寸 (width, height)
        interpolation=cv2.INTER_LINEAR)    # 双线性插值
    
    
    # 3. 创建灰度图像（反转：1表示材料，0表示空隙）
    gray_image = density_data
    
    
    
    # 4. 创建3×3拼接图像
    density_data_9 = np.tile(gray_image, (3, 3))
    
    # plt.figure()
    # plt.imshow(density_data_9, cmap='gray')
    
    # 5. 应用更强的模糊处理（增加sigma值）
    smoothed_density = gaussian_filter(density_data_9, sigma=10.0, mode='wrap')  # 增加sigma值
    
    # plt.figure()
    # plt.imshow(smoothed_density, cmap='gray')
    
    # 6. 裁剪中心区域
    cut_density = smoothed_density[990:2010, 990:2010] 
    
    plt.figure()
    plt.imshow(cut_density, cmap='gray')
    
    # 7. 转换为8位图像
    cut_density_8bit = (cut_density * 255).astype(np.uint8)
    _, binary_image1 = cv2.threshold(cut_density_8bit, 127, 255, cv2.THRESH_BINARY)
    
    plt.figure()
    plt.imshow(binary_image1, cmap='gray')
    
    
    # 9. 应用形态学操作平滑边界（闭运算填充小孔，开运算去除小噪点）
    kernel = np.ones((3, 3), np.uint8)
    smoothed_binary = cv2.morphologyEx(binary_image1, cv2.MORPH_CLOSE, kernel)
    smoothed_binary = cv2.morphologyEx(smoothed_binary, cv2.MORPH_OPEN, kernel)
    
    # 10. 查找轮廓（只检测最外层轮廓）
    contours, hierarchy = cv2.findContours(
        smoothed_binary, 
        cv2.RETR_EXTERNAL,  # 只检测外层轮廓
        cv2.CHAIN_APPROX_NONE)
    output_path = os.path.join(contour_path,"test.txt")
    f=open(output_path,'w')
    
    for i in range(len(contours)):#
        contour_data=contours[i].squeeze(axis=1).tolist()

        for j in range(len(contour_data)):
            x, y = contour_data[j][0], contour_data[j][1]
            f.write(str(x/(1000.0/30.0))+'\t')
        f.write('\n')
        
        for j in range(len(contour_data)):
            x, y = contour_data[j][0], contour_data[j][1]
            f.write(str(y/(1000.0/30.0))+'\t')
        f.write('\n')
    
    f.close()

    print('contour save in '+ output_path)



if __name__=='__main__':
    import sys
    if len(sys.argv) != 3:
        print("用法: python LLM_CAE_FE_1.py <image_path> <contour_path>")
        sys.exit(1)
    image_path = sys.argv[1]
    contour_path = sys.argv[2]
    from_image_to_contour(image_path, contour_path)
