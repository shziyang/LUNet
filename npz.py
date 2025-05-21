import glob
import numpy as np
import cv2

def npz():
#图像路径
    path = r'E:\AI model\JPG-npz\train\images\*.png'
    #项目中存放训练所用的npz文件路径
    path2 = r'E:\AI model\JPG-npz\UAV\\'
    for i,img_path in enumerate(glob.glob(path)):
        #读入图像
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        #读入标签
        label_path = img_path.replace('images', 'labels')
        # label_path = label_path.replace('jpg', 'png')
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        image = np.array(image)
        label = np.array(label)

        #保存npz
        np.savez(path2+str(1055+i),image=image,label=label)
        print('------------',i)

    #加载npz文件
    data = np.load(r'E:\AI model\JPG-npz\lables two\0.npz', allow_pickle=True)
    image, label = data['image'], data['label']
    cv2.imshow('Loaded Image', image)
    cv2.waitKey(0)  # 等待按键
    cv2.destroyAllWindows()  # 关闭所有窗口

    # 如果标签是二值图像，你可能需要先将其转换为可视化格式
    # 例如，使用颜色映射来显示分割结果
    label_colored = cv2.applyColorMap(label, cv2.COLORMAP_JET)
    cv2.imshow('Loaded Label', label_colored)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print('ok')

npz()