'''
用于可视化 .npy 文件的图像
'''
import numpy as np
import matplotlib.pyplot as plt

def visualize_npy_image_advanced(npy_path):
    """高级版本的 .npy 图像可视化"""
    # 加载数据
    img = np.load(npy_path)
    
    # 打印基本信息
    print("=== 图像信息 ===")
    print(f"形状: {img.shape}")
    print(f"类型: {img.dtype}")
    print(f"范围: [{img.min()}, {img.max()}]")
    print(f"均值: {img.mean():.4f}")
    print(f"标准差: {img.std():.4f}")
    
    # 创建图像网格
    fig = plt.figure(figsize=(15, 5))
    
    # 1. 原始图像
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
    
    # # 2. 热力图
    # plt.subplot(132)
    # if len(img.shape) == 3:
    #     # 如果是多通道，显示平均值
    #     plt.imshow(np.mean(img, axis=0), cmap='hot')
    #     plt.title('Channel Mean (Heatmap)')
    # else:
    #     plt.imshow(img, cmap='hot')
    #     plt.title('Heatmap')
    # plt.colorbar()
    
    # # 3. 3D表面图
    # plt.subplot(133)
    # if len(img.shape) == 3:
    #     data_2d = np.mean(img, axis=0)
    # else:
    #     data_2d = img
    # X, Y = np.meshgrid(np.arange(data_2d.shape[1]), np.arange(data_2d.shape[0]))
    # ax = fig.add_subplot(133, projection='3d')
    # ax.plot_surface(X, Y, data_2d, cmap='viridis')
    # ax.set_title('3D Surface Plot')
    
    # plt.tight_layout()
    # plt.show()

# 使用示例
if __name__ == "__main__":
    npy_path = '../temp_obs/temp_save_0_2_image.npy'
    visualize_npy_image_advanced(npy_path)


