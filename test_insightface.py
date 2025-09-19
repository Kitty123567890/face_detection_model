# 1安装必要的库
# pip install insightface
# pip install onnxruntime-gpu # 有NVIDIA GPU并安装了CUDA
# 或者
# pip install onnxruntime # 只使用CPU

import insightface
from insightface.app import FaceAnalysis
import cv2
import numpy as np

# 初始化 FaceAnalysis，并指定模型包名称
# providers: 指定计算后端。'CUDAExecutionProvider' 使用GPU, 'CPUExecutionProvider' 使用CPU。
#可选预训练模型包括:
#   Name	    Detection Model	    Recognition Model	    Alignment	    Attributes	Model-Size
#   antelopev2	RetinaFace-10GF	    ResNet100@Glint360K	    2d106 & 3d68	Gender&Age	407MB
#   buffalo_l	RetinaFace-10GF	    ResNet50@WebFace600K	2d106 & 3d68	Gender&Age	326MB
#   buffalo_m	RetinaFace-2.5GF	ResNet50@WebFace600K	2d106 & 3d68	Gender&Age	313MB
#   buffalo_s	RetinaFace-500MF	MBF@WebFace600K	        2d106 & 3d68	Gender&Age	159MB
#   buffalo_sc	RetinaFace-500MF	MBF@WebFace600K	        -	            -	        16MB
try:
    modelname = 'buffalo_l'  # 选择模型名称
    app = FaceAnalysis(name=modelname, 
                       providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    # 准备模型，ctx_id指定使用的设备ID (0代表第一块GPU，-1代表CPU)
    # det_size 是检测模型的输入图像尺寸，可以根据需要调整
    app.prepare(ctx_id=0, det_size=(640, 640))
    print("模型加载成功:", modelname)
except Exception as e:
    print(f"模型加载失败，请检查环境或模型名称: {e}")
    exit()


# 3. 读取图片并进行人脸分析
img_path = './test_image.jpg' # 替换成你的图片路径
img = cv2.imread(img_path)

if img is None:
    print(f"无法读取图片: {img_path}")
    exit()

# get() 函数会完成检测、关键点提取、对齐、特征提取所有步骤
faces = app.get(img)

# 4. 处理和输出结果
if not faces:
    print("在图片中没有检测到人脸。")
else:
    print(f"在图片中检测到 {len(faces)} 张人脸。")
    for i, face in enumerate(faces):
        print(f"\n--- 人脸 {i+1} ---")
        
        # 边界框 (bounding box)
        bbox = face.bbox.astype(int)
        print(f"  - 位置 (边界框): [x1, y1, x2, y2] = {bbox}")
        
        # 人脸特征向量 (embedding)
        embedding = face.embedding
        print(f"  - 特征向量 (Embedding): shape={embedding.shape}, dtype={embedding.dtype}")
        # 这是一个512维的numpy数组，可以用于后续的身份比对
        # print(embedding)可以查看向量内容
        
        # 其他信息
        if face.kps is not None:
            print(f"  - 关键点 (Keypoints) 数量: {len(face.kps)}")
        if face.sex is not None:
            print(f"  - 性别: {face.sex}, 年龄: {face.age}") # buffalo_l 默认不包含性别年龄模型
            
        # 可视化：在图上画出边界框
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

    cv2.imwrite("./output_image.jpg", img)
    print("\n结果已保存为 output_image.jpg")