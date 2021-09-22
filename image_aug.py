
from PIL import Image, ImageEnhance, ImageOps, ImageFile,ImageChops,ImageFont,ImageDraw
import numpy as np
import random
import threading, os, time
import logging
import math
import cv2
import xml
import xml.dom.minidom

logger = logging.getLogger(__name__)
ImageFile.LOAD_TRUNCATED_IMAGES = True
global count
count = 0

def genXml(xml_info,xml_path):

    #create an empty dom document object
    doc = xml.dom.minidom.Document()
    #creat a root node which name is annotation
    annotation = doc.createElement('annotation')
    #add the root node to the dom document object
    doc.appendChild(annotation)

    #add the folder subnode
    folder = doc.createElement('folder')
    folder_text = doc.createTextNode('images')
    folder.appendChild(folder_text)
    annotation.appendChild(folder)

    #add the filename subnode
    img_file = xml_info["filename"]
    filename = doc.createElement('filename')
    filename_text = doc.createTextNode(img_file)
    filename.appendChild(filename_text)
    annotation.appendChild(filename)

    # add the path subnode
    path = doc.createElement('path')
    path_text = doc.createTextNode('/data/haoyuan/YOLOX/datasets/VOCdevkit/VOC2007/JPEGImages' + img_file)
    path.appendChild(path_text)
    annotation.appendChild(path)

    #add the source subnode
    source = doc.createElement('source')
    database = doc.createElement('database')
    database_text = doc.createTextNode('Unknown')
    source.appendChild(database)
    database.appendChild(database_text)
    annotation.appendChild(source)

    #add the size subnode
    size = doc.createElement('size')
    width = doc.createElement('width')
    width_text = doc.createTextNode(str(xml_info["width"]))
    height = doc.createElement('height')
    height_text = doc.createTextNode(str(xml_info["height"]))
    depth = doc.createElement('depth')
    depth_text = doc.createTextNode('3')
    size.appendChild(width)
    width.appendChild(width_text)
    size.appendChild(height)
    height.appendChild(height_text)
    size.appendChild(depth)
    depth.appendChild(depth_text)
    annotation.appendChild(size)

    #add the segmented subnode
    segmented = doc.createElement('segmented')
    segmented_text = doc.createTextNode('0')
    segmented.appendChild(segmented_text)
    annotation.appendChild(segmented)

    #add the object subnode
    obj = doc.createElement('object')
    name = doc.createElement('name')
    name_text = doc.createTextNode(str(xml_info["name"]))

    pose = doc.createElement('pose')
    pose_text = doc.createTextNode("Unspecified")

    truncated = doc.createElement('truncated')
    truncated_text = doc.createTextNode(str(0))

    difficult = doc.createElement('difficult')
    difficult_text = doc.createTextNode(str(0))

    bndbox = doc.createElement('bndbox')

    xmin = doc.createElement('xmin')
    ymin = doc.createElement('ymin')
    xmax = doc.createElement('xmax')
    ymax = doc.createElement('ymax')
    xmin_text = doc.createTextNode(str(xml_info["xmin"]))
    ymin_text = doc.createTextNode(str(xml_info["ymin"]))
    xmax_text = doc.createTextNode(str(xml_info["xmax"]))
    ymax_text = doc.createTextNode(str(xml_info["ymax"]))

    obj.appendChild(name)
    name.appendChild(name_text)

    obj.appendChild(pose)
    pose.appendChild(pose_text)

    obj.appendChild(truncated)
    truncated.appendChild(truncated_text)

    obj.appendChild(difficult)
    difficult.appendChild(difficult_text)

    obj.appendChild(bndbox)
    bndbox.appendChild(xmin)
    xmin.appendChild(xmin_text)
    bndbox.appendChild(ymin)
    ymin.appendChild(ymin_text)
    bndbox.appendChild(xmax)
    xmax.appendChild(xmax_text)
    bndbox.appendChild(ymax)
    ymax.appendChild(ymax_text)


    annotation.appendChild(obj)

    #write into the xml text file
    # os.mknod(xml_path+'%s.xml'%img_name)
    print(img_file)
    fp = open(os.path.join(xml_path,'%s.xml'%img_file.split(".jpg")[0]), 'w+')
    doc.writexml(fp, indent='\t', addindent='\t', newl='\n')
    fp.close()

class DataAugmentation:

    def __init__(self):
        pass

    @staticmethod
    def openImage(image):
        return Image.open(image)

    @staticmethod
    def composeImage(image,background,xml_info):
        """标定图像image贴到background"""
        
        width,height = background.size
        w,h = image.size
        random_size = random.uniform(0.1,0.5)  #相对背景图resize比例因子

        if width<height:
            width_trans = width
            height_trans = np.min((height,int((width/w)*h)))        
        else:
            height_trans = height
            width_trans = np.min((width,int((height/h)*w)))

        width_trans *= random_size
        height_trans *= random_size
        width_trans,height_trans = int(width_trans),int(height_trans)
        image = image.resize((width_trans,height_trans)) #根据背景图大小调整image
        
        
        pos = (np.random.randint(0,width-width_trans),np.random.randint(0,height-height_trans))     #随机贴图位置
        if image.mode == "RGBA":
            A = image.split()[3]
            background.paste(im = image,box=pos,mask = A)
            width_trans = np.max(np.where(A)[1]) - np.min(np.where(A)[1])
            height_trans = np.max(np.where(A)[0]) - np.min(np.where(A)[0])
        else:
            background.paste(im = image,box=pos)
        xml_info["width"] = width
        xml_info["height"] = height
        xml_info["xmin"] = pos[0]
        xml_info["ymin"] = pos[1]
        xml_info["xmax"] = pos[0]+width_trans
        xml_info["ymax"] = pos[1]+height_trans
        

        return background,xml_info

    @staticmethod
    def randomText(image):
        w,h = image.size
        text = ''
        len_text = random.randint(0,15)
        for i in range(len_text):
            val = random.randint(0x4e00, 0x9fbf)
            text+=chr(val)
        for i in range(random.randint(0,5)):
            size = random.randint(10,w//5)
            setFont = ImageFont.truetype('simsun.ttf', size)

            pos = (random.randint(0,w-size),random.randint(0,h))
            fillColor = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
            draw = ImageDraw.Draw(image)
            draw.text(pos,text,font=setFont,fill=fillColor,direction=None)
        return image


    @staticmethod
    def randomRotation(image, mode=Image.BICUBIC):
        """ 对图像进行随机任意角度(0~360度)旋转 :param mode 邻近插值,双线性插值,双三次B样条插值(default)"""
        random_angle = np.random.randint(-45, 45)
        # random_angle = np.random.choice([0,90,180,270])
        return image.rotate(random_angle, mode)

    @staticmethod
    def randomCrop(image):
        """ 对图像随意剪切"""
        image_width = image.size[0]
        image_height = image.size[1]
        
        crop_win_size = np.random.randint(min(image_height,image_width)//2, min(image_height,image_width))
        random_region = (
            (image_width - crop_win_size) >> 1, (image_height - crop_win_size) >> 1, (image_width + crop_win_size) >> 1,
            (image_height + crop_win_size) >> 1)
        return image.crop(random_region)


    @staticmethod
    def randomFlip(image, mode=Image.FLIP_LEFT_RIGHT):
        """
        对图像进行上下左右四个方面的随机翻转
        :param model: 水平或者垂直方向的随机翻转模式,默认右向翻转
        """
        random_model = np.random.randint(0, 4)
        if random_model>1:
            return image
        flip_model = [Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM]
        return image.transpose(flip_model[random_model])
        #return image.transpose(mode)

    @staticmethod
    def randomShift(image):
    #def randomShift(image, xoffset, yoffset=None):
        """
        对图像进行平移操作
        :param xoffset: x方向向右平移
        :param yoffset: y方向向下平移
        """
        random_xoffset = np.random.randint(0, math.ceil(image.size[0]*0.2))
        random_yoffset = np.random.randint(0, math.ceil(image.size[1]*0.2))
        return ImageChops.offset(image,xoffset = random_xoffset, yoffset = random_yoffset)
        # return image.offset(random_xoffset)

    @staticmethod
    def randomJitter(image):
        """ 对图像进行色彩抖动"""
        random_factor = np.random.randint(5, 31) / 20.  # 随机因子
        color_image = ImageEnhance.Color(image).enhance(random_factor)  # 调整图像的饱和度

        random_factor = np.random.randint(5, 15) / 10.  # 随机因子
        brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)  # 调整图像的亮度
        
        random_factor = np.random.randint(5, 30) / 20.  # 随机因子
        contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)  # 调整图像对比度

        random_factor = np.random.randint(0, 31) / 20.  # 随机因子
        return ImageEnhance.Sharpness(contrast_image).enhance(random_factor)  # 调整图像锐度
    
    @staticmethod
    def randomColor(image):
        """ 随机改变颜色"""
        image = image.convert('RGB')
        t = random.randint(1,5)
        r, g, b = image.split()
        RGB  = [r,g,b]
        print(t)
        #纯色
        if t>=4:
            for i in range(3):
                color = random.randint(0,255)
                RGB[i] = RGB[i].point(lambda i: color)
        #随机通道变换
        if t<4:           
            for i in range(t):
                color = random.randint(0,255)
                channel = random.randint(0,2)               
                RGB[channel] = RGB[channel].point(lambda i: color)
        color_img = Image.merge('RGB', tuple(RGB))        

        #渐变色
        if t==5:
            w,h = image.size
            result = np.zeros((w, h, 3), dtype=np.uint8)
            rgb_start = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
            rgb_stop = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
            for i, (m,n,o) in enumerate(zip(rgb_start, rgb_stop, (random.choice([True, False]),random.choice([True, False]),random.choice([True, False])))):
                if o:
                    result[:,:,i] = np.tile(np.linspace(m, n, w), (h, 1))
                else:
                    result[:,:,i] = np.tile(np.linspace(m, n, w), (h, 1)).T
            color_img = Image.fromarray(result)
        #反色
        if t <3:
            if random.randint(0,2)==2:
                color_img = ImageChops.invert(color_img)
        return  color_img
    
                             
    
    @staticmethod
    def randomGaussian(image, mean=0.2, sigma=0.3):
        """ 对图像进行高斯噪声处理"""

        def gaussianNoisy(im, mean=0.2, sigma=0.3):
            """ 对图像做高斯噪音处理 :param im: 单通道图像 :param mean: 偏移量 :param sigma: 标准差 :return: """
            for _i in range(len(im)):
                im[_i] += random.gauss(mean, sigma)
            return im

        # 将图像转化成数组
        img = np.array(image)
        img.flags.writeable = True  # 将数组改为读写模式
        width, height = img.shape[:2]
        img_r = gaussianNoisy(img[:, :, 0].flatten(), mean, sigma)
        img_g = gaussianNoisy(img[:, :, 1].flatten(), mean, sigma)
        img_b = gaussianNoisy(img[:, :, 2].flatten(), mean, sigma)
        img[:, :, 0] = img_r.reshape([width, height])
        img[:, :, 1] = img_g.reshape([width, height])
        img[:, :, 2] = img_b.reshape([width, height])
        return Image.fromarray(np.uint8(img))

    @staticmethod
    def randomPerspective(image):
        """ 对图像进行透视变换 """
        w,h = image.size
        a_image = np.array(image)
        r = [random.uniform(0.05,0.15) for i in range(4)]    #透视变换随机因子
        pts_ori = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]])   #原坐标
        pts_dst = np.float32([[w*r[0],h*r[0]],[w*r[1],h-h*r[1]],[w-w*r[2],h-h*r[2]],[w-w*r[3],h*r[3]]])   #变换后坐标
        M = cv2.getPerspectiveTransform(pts_ori,pts_dst)

        img_trans = cv2.warpPerspective(a_image,M,(w,h))

        return Image.fromarray(img_trans)

    @staticmethod
    def saveImage(image, path,mode='RGB'):
        image = image.convert(mode)
        image.save(path)


def makeDir(path):
    try:
        if not os.path.exists(path):
            if not os.path.isfile(path):
                # os.mkdir(path)
                os.makedirs(path)
            return 0
        else:
            return 1
    except Exception as  e:
        print(str(e))
        return -2


def imageOps(ops_list, image,des_path, file_name, xml_info,times=1):
    global count
    # funcMap = {"Rotation": DataAugmentation.randomRotation,
    #            "Crop": DataAugmentation.randomCrop,
    #            "Jitter": DataAugmentation.randomJitter,
    #            "Color": DataAugmentation.randomColor,
    #            "Gaussian": DataAugmentation.randomGaussian,
    #            'Shift':DataAugmentation.randomShift,
    #            'Flip':DataAugmentation.randomFlip
    #            }

    
    img_name = "%05d"%count+file_name+'.jpg'
    xml_info["filename"] = img_name
    DataAugmentation.saveImage(image, os.path.join(des_path, img_name))
    # genXml(xml_info,"/Users/qline/Documents/imgo/ExamineVerify/detect_data/flag_choose_/label")
    count+=1

ops_list = [ "Jitter", "Gaussian"]


def threadOPS(img_path,bg_path,new_path):
    """ 多线程处理事务 :param src_path: 资源文件 :param des_path: 目的地文件 :return: """
    if os.path.isdir(img_path):
        img_names = []
        img_paths = []
        for root,dirs,files in os.walk(img_path):
            for img_name in files:
                if '.DS_Store' not in img_name:
                    img_names.append(img_name)
                    img_paths.append(os.path.join(root,img_name))

    else:
        img_names = [img_path]
    bg_names = os.listdir(bg_path)
    if '.DS_Store' in bg_names:
        bg_names.remove('.DS_Store')

    for img_name,cur_img_path in zip(img_names,img_paths):
        #print(img_name)
        xml_info = {}
        xml_info["folder"] = cur_img_path.split('/')[-2]
        xml_info["name"] = cur_img_path.split('/')[-2]
        image = DataAugmentation.openImage(cur_img_path)
        for bg_name in bg_names:
            file_name = '_cmp'
            cur_bg_path = os.path.join(bg_path, bg_name)

            background = DataAugmentation.openImage(cur_bg_path)
            #裁剪出地图区域
            
            #四分之一的几率对地图部分做透视变换
            # rand_compose = np.random.randint(4)
            # if rand_compose==0:     
            #     file_name = '_warp'                                             
            #     image = DataAugmentation.randomPerspective(image)
            # elif rand_compose==1:
            #     file_name = '_rota'                                             
            #     image = DataAugmentation.randomRotation(image)

            #随机对背景做处理
            rand_compose = np.random.randint(4)
            if rand_compose == 0:
                file_name = '_shift'
                background = DataAugmentation.randomShift(background)
                
            elif rand_compose == 1:
                file_name = '_crop'
                background = DataAugmentation.randomCrop(background)
            
            elif rand_compose == 2:
                file_name = '_color'
                # background = DataAugmentation.randomColor(background)
                background = DataAugmentation.randomJitter(background)

            background = DataAugmentation.randomFlip(background)

            # #贴图到背景
            # image_trans,xml_info = DataAugmentation.composeImage(image,background,xml_info)                    
            
            # threadImage = [0]
            # _index = 0
            try:
                imageOps(ops_list, background,new_path, file_name,xml_info)
            except Exception as e:
                print(e)
                continue
                # threadImage[0] = threading.Thread(target=imageOps,
                #                                     args=(ops_name, image_trans,mask_trans,new_path, file_name))
                # threadImage[0].start()
                # _index += 1
                # time.sleep(0.2)



if __name__ == '__main__':
    threadOPS("/data/haoyuan/sta",          #检测图像
              "/data/haoyuan/testttt",      #背景
              "/data/haoyuan/xie_Data",     #保存路径
              )

