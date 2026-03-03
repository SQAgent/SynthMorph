import argparse
from PIL import Image
import os

def images_to_gif(image_files, output_gif="output.gif", duration=500, loop=0):

    try:
        # 打开所有图片
        images = [Image.open(img) for img in image_files]
        
        # 保存为GIF
        images[0].save(
            output_gif,
            save_all=True,
            append_images=images[1:],
            duration=duration,
            loop=loop,
            optimize=True
        )
        
        print(f"GIF已成功创建：{output_gif}")
        print(f"包含 {len(images)} 张图片，每帧 {duration} 毫秒")
        
        # 显示GIF大小
        gif_size = os.path.getsize(output_gif)
        print(f"GIF文件大小：{gif_size / 1024:.2f} KB")
        
    except Exception as e:
        print(f"创建GIF时出错：{e}")

if __name__ == "__main__":
    
    
    image_path = "/home/shangqing/sqdata/project/sqagents/src/sqagents/StructurePrediction/Tools/generation_process/"
    image_files = [image_path+"step_0000_sample_001.png",
                   image_path+"step_0100_sample_001.png",
                   image_path+"step_0200_sample_001.png",
                   image_path+"step_0300_sample_001.png",
                   image_path+"step_0400_sample_001.png",
                   image_path+"step_0500_sample_001.png",
                   image_path+"step_0600_sample_001.png",
                   image_path+"step_0700_sample_001.png",
                   image_path+"step_0800_sample_001.png",
                   image_path+"step_0900_sample_001.png"]
    
    # 创建GIF
    images_to_gif(
        image_files=image_files,
        output_gif="animation.gif",
        duration=300,  # 每帧300毫秒
        loop=0  # 无限循环
        )