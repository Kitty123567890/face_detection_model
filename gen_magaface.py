from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import sys
import numpy as np
import argparse
import struct
import cv2
import sklearn
from sklearn.preprocessing import normalize
from FaceRecognizer import FaceRecognizer  # 导入FaceRecognizer类

def read_img(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    return img

def get_feature(imgs, recognizer):
    """
    使用FaceRecognizer处理图像并提取特征
    """
    features = []
    for img in imgs:
        # 使用FaceRecognizer提取人脸特征
        faces = recognizer.recognize(img)
        
        if not faces:
            # 如果没有检测到人脸，创建一个全零向量
            embedding = np.zeros(512, dtype=np.float32)
            print("警告: 未检测到人脸，使用零向量")
        else:
            # 选择最大的人脸（按边界框面积）
            if len(faces) > 1:
                faces = sorted(faces, key=lambda face: (face.bbox[2]-face.bbox[0])*(face.bbox[3]-face.bbox[1]), reverse=True)
            face = faces[0]
            embedding = face.embedding
        
        # 确保特征向量是512维
        if embedding.shape[0] != 512:
            print(f"警告: 特征维度不是512，实际为{embedding.shape[0]}")
            # 如果维度不对，创建一个全零向量
            embedding = np.zeros(512, dtype=np.float32)
        
        features.append(embedding)
    
    # 转换为numpy数组并归一化
    features = np.array(features)
    features = normalize(features)
    return features

def write_bin(path, feature):
    feature = list(feature)
    with open(path, 'wb') as f:
        f.write(struct.pack('4i', len(feature), 1, 4, 5))
        f.write(struct.pack("%df" % len(feature), *feature))

def get_and_write(buffer, recognizer):
    imgs = [k[0] for k in buffer]
    features = get_feature(imgs, recognizer)
    
    assert features.shape[0] == len(buffer)
    for ik, k in enumerate(buffer):
        out_path = k[1]
        feature = features[ik].flatten()
        write_bin(out_path, feature)

def main(args):
    print(args)
    
    # 初始化FaceRecognizer
    recognizer = FaceRecognizer(
        model_name=args.algo, 
        ctx_id=args.gpu, 
        det_size=(args.det_size, args.det_size) if hasattr(args, 'det_size') else (640, 640)
    )
    
    facescrub_out = os.path.join(args.output, 'facescrub')
    megaface_out = os.path.join(args.output, 'megaface')

    # 处理FaceScrub数据集
    i = 0
    succ = 0
    buffer = []
    start_time = time.time()
    for line in open(args.facescrub_lst, 'r'):
        if i % 1000 == 0:
            print("writing fs", i, succ)
        i += 1
        image_path = line.strip()
        _path = image_path.split('/')
        a, b = _path[-2], _path[-1]
        out_dir = os.path.join(facescrub_out, a)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        image_path = os.path.join(args.facescrub_root, image_path)
        img = read_img(image_path)
        if img is None:
            print('read error:', image_path)
            continue
        out_path = os.path.join(out_dir, b + "_%s.bin" % args.algo)
        item = (img, out_path)
        buffer.append(item)
        if len(buffer) == args.batch_size:
            get_and_write(buffer, recognizer)
            buffer = []
        succ += 1
    if len(buffer) > 0:
        get_and_write(buffer, recognizer)
        buffer = []
    end_time = time.time()
    print("FaceScrub处理时间: {:.2f}秒".format(end_time - start_time))
    print('fs stat', i, succ)
    
    if args.nomf:
        return

    # 处理MegaFace数据集
    i = 0
    succ = 0
    buffer = []
    start_time = time.time()
    for line in open(args.megaface_lst, 'r'):
        if i % 1000 == 0:
            print("writing mf", i, succ)
        i += 1
        image_path = line.strip()
        _path = image_path.split('/')
        a1, a2, b = _path[-3], _path[-2], _path[-1]
        out_dir = os.path.join(megaface_out, a1, a2)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        image_path = os.path.join(args.megaface_root, image_path)
        img = read_img(image_path)
        if img is None:
            print('read error:', image_path)
            continue
        out_path = os.path.join(out_dir, b + "_%s.bin" % args.algo)
        item = (img, out_path)
        buffer.append(item)
        if len(buffer) == args.batch_size:
            get_and_write(buffer, recognizer)
            buffer = []
        succ += 1
    if len(buffer) > 0:
        get_and_write(buffer, recognizer)
        buffer = []
    end_time = time.time()
    print("MegaFace处理时间: {:.2f}秒".format(end_time - start_time))
    print('mf stat', i, succ)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--batch_size', type=int, help='', default=8)
    parser.add_argument('--det_size', type=int, help='Detection size', default=640)
    parser.add_argument('--gpu', type=int, help='', default=0)
    parser.add_argument('--algo', type=str, help='', default='buffalo_l')
    parser.add_argument('--facescrub-lst', type=str, help='', default='./data/facescrub_lst')
    parser.add_argument('--megaface-lst', type=str, help='', default='./data/megaface_lst')
    parser.add_argument('--facescrub-root', type=str, help='', default='./data/facescrub_images')
    parser.add_argument('--megaface-root', type=str, help='', default='./data/megaface_images')
    parser.add_argument('--output', type=str, help='', default='./feature_out')
    parser.add_argument('--nomf', default=False, action="store_true", help='')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))