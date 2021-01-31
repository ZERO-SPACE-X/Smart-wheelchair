# from PIL import Image
#
# txt_path = './labels.txt'
#
# with open(txt_path, 'r') as f:
#     img_nums = []
#     for line in f:
#         image_list = [img_path for img_path in line.split(',')]
#         img_nums.append((image_list[:-1],image_list[-1]))
# print(img_nums[0][0][0])
# image = Image.open(img_nums[0][0][0]).convert('RGB')
# print(image)

import visdom
import torch
# 新建一个连接客户端
# 指定env = 'test1'，默认是'main',注意在浏览器界面做环境的切换
vis = visdom.Visdom(env='test1')
# 绘制正弦函数
x = torch.arange(1, 100, 0.01)
y = torch.sin(x)
vis.line(X=x,Y=y, win='sinx',opts={'title':'y=sin(x)'})
# 绘制36张图片随机的彩色图片
vis.images(torch.randn(36,3,64,64).numpy(),nrow=6, win='imgs',opts={'title':'imgs'})




# print(image_list)
# print(img_nums)
# print(len(img_nums))