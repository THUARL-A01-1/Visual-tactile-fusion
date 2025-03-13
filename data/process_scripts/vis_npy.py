'''
用于可视化 .npy 文件的图像
'''
import numpy as np
import matplotlib.pyplot as plt

def visualize_npy_image_advanced(npy_path):
    img = np.load(npy_path)
    
    # 打印基本信息
    print("=== img info ===")
    print(f"shape: {img.shape}")
    print(f"dtype: {img.dtype}")
    print(f"range: [{img.min()}, {img.max()}]")
    print(f"mean: {img.mean():.4f}")
    print(f"std: {img.std():.4f}")
    
    fig = plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    if len(img.shape) == 3 and img.shape[0] == 3:
        # RGB图像，CHW -> HWC
        plt.imshow(img.transpose(1, 2, 0))
        plt.title('RGB Image')
    else:
        # 灰度图或其他
        plt.imshow(img, cmap='viridis')
        plt.title('Original Image')
    plt.colorbar()
    save_path = npy_path.replace('.npy', '.png')
    plt.savefig(save_path)
    print(f"Saved image to {save_path}")



if __name__ == "__main__":
    npy_path = '../temp_obs/temp_save_0_2_image.npy'
    visualize_npy_image_advanced(npy_path)


