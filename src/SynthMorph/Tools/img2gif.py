import argparse
from PIL import Image
import os

def images_to_gif(image_files, output_gif="output.gif", duration=500, loop=0):

    try:
        # Open all images
        images = [Image.open(img) for img in image_files]
        
        # Save as GIF
        images[0].save(
            output_gif,
            save_all=True,
            append_images=images[1:],
            duration=duration,
            loop=loop,
            optimize=True
        )
        
        print(f"GIF successfully created: {output_gif}")
        print(f"Contains {len(images)} images, {duration} ms per frame")
        
        # Show GIF size
        gif_size = os.path.getsize(output_gif)
        print(f"GIF file size: {gif_size / 1024:.2f} KB")
        
    except Exception as e:
        print(f"Error creating GIF: {e}")

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
    
    # Create GIF
    images_to_gif(
        image_files=image_files,
        output_gif="animation.gif",
        duration=300,  # 300 ms per frame
        loop=0  # Infinite loop
        )