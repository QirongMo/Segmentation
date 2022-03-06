
# %matplotlib inline
import paddle
import cv2
import matplotlib.pyplot as plt
import numpy as np

def show_image_color(color_txt,image):
    f = open(color_txt, 'r')
    colors = f.readlines()
    h, w= image.shape[:2]
    for i in range(h):
        for j in range(w):
            class_id = int(image[i, j, 0].item())
            color = colors[class_id].strip().split()
            color = list(map(int, color))
            image[i, j, :] = color
    f.close()
    plt.imshow(image/255.0)
    plt.axis('off')  # 关掉坐标轴为 off
    plt.title('label')  # 图像题目
    plt.show()
    return image

def show_grayimage_color(color_txt,image):
    f = open(color_txt, 'r')
    colors = f.readlines()
    h, w= image.shape[:2]
    showimage = np.zeros((h,w,3))
    for i in range(h):
        for j in range(w):
            class_id = int(image[i, j].item())
            class_id = int(class_id)
            color = colors[class_id].strip().split()
            color = list(map(int, color))
            showimage[i, j, :] = color
    f.close()
    plt.imshow(showimage/255.0)
    plt.axis('off')  # 关掉坐标轴为 off
    plt.title('predict')  # 图像题目
    plt.show()

    return showimage

def merge(path, img):
    image = cv2.imread(path,  cv2.COLOR_BGR2RGB)
    img = np.uint8(img)
    overlapping = cv2.addWeighted(image, 0.6, img, 0.4, 0)
    plt.imshow(overlapping/255.0)
    plt.title('merge')
    plt.show()

def main():
    size = 512
    color_txt = './dataset/celeb_colors.txt'
    # load model
    from model.UNet import UNet
    model = UNet(19)
    # 加载模型权重
    param_dict = paddle.load('./weight/UNet/UNet.pdparams')
    model.load_dict(param_dict)
    model.eval()

    #load original image
    # img_path = './dataset/CelebAMask-HQ/CelebA-HQ-img/5000.jpg'
    img_path = './5000.jpg'
    img = cv2.imread(img_path)
    h,w = img.shape[:2]
    img = cv2.resize(img,(size,size),interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)/255.0
    img = img[np.newaxis,:,:,:].astype(np.float32)
    img = paddle.to_tensor(img).transpose((0,3,1,2))
    # predict
    pred = model(img)
    pred = paddle.nn.Softmax(axis=1)(pred)
    pred = np.array(pred.numpy())
    pred = np.argmax(pred, axis=1)
    pred = pred.reshape(size, size).astype(np.uint8)
    # show
    show_grayimage_color(color_txt, pred)
    pred = cv2.resize(pred, (h,w))
    pred_image = show_grayimage_color(color_txt,pred)
    # merge predict and origion img
    merge(img_path, pred_image)

if __name__ == "__main__":
    main()