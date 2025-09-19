import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
import numpy as np
import os
import cv2
from pathlib import Path

# #########################
# 1. 模型初始化与配置
# #########################

# 选择预训练模型配置
# 可用模型: 'buffalo_l', 'buffalo_s', 'antelopev2', 'antelopev2_glint' 等
# 'buffalo_l' - 大型模型，精度最高
# 'buffalo_s' - 小型模型，速度更快
model_name = 'buffalo_l'  # 根据需求在精度和速度间权衡

# 初始化FaceAnalysis应用
app = FaceAnalysis(
    name=model_name,      # 选择模型
    root='~/.insightface', # 模型下载路径
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider'] # 优先使用GPU， fallback到CPU
)
app.prepare(ctx_id=0, det_size=(640, 640))  # ctx_id=0 表示使用GPU 0, -1 表示CPU

# #########################
# 2. 构建人脸注册库
# #########################

# 已知人脸的数据库 {identity_name: embedding_vector}
known_faces_db = {}

def build_face_database(database_dir):
    """
    构建人脸特征数据库
    :param database_dir: 包含已知人脸图像的文件夹路径
                        建议每张图片以'人物姓名.jpg'格式命名
    """
    database_path = Path(database_dir)
    if not database_path.exists():
        print(f"数据库路径 {database_dir} 不存在！")
        return
    
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    
    for image_path in database_path.glob('*'):
        if image_path.suffix.lower() not in image_extensions:
            continue
            
        # 从文件名提取身份ID（去掉扩展名）
        identity_id = image_path.stem
        
        # 读取图像
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"无法读取图像: {image_path}")
            continue
            
        # 检测人脸并提取特征
        faces = app.get(img)
        
        if len(faces) == 0:
            print(f"在 {image_path} 中未检测到人脸")
            continue
        elif len(faces) > 1:
            print(f"警告: 在 {image_path} 中检测到多张人脸，将使用第一张")
        
        # 获取第一个人脸的特征向量
        face = faces[0]
        embedding = face.normed_embedding  # 归一化后的特征向量
        
        # 存入数据库
        known_faces_db[identity_id] = embedding
        print(f"已注册: {identity_id}")
    
    print(f"\n注册完成！共注册 {len(known_faces_db)} 个身份")

# #########################
# 3. 人脸识别函数
# #########################

def recognize_face(image_path, threshold=0.6):
    """
    识别单张图像中的人脸
    :param image_path: 待识别图像路径
    :param threshold: 相似度阈值，低于此值认为是未知人员
    :return: 识别结果列表
    """
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图像: {image_path}")
        return []
    
    # 检测人脸
    faces = app.get(img)
    
    results = []
    
    for i, face in enumerate(faces):
        # 获取人脸信息
        bbox = face.bbox.astype(int)  # 边界框 [x1, y1, x2, y2]
        confidence = face.det_score    # 检测置信度
        landmarks = face.kps           # 5个关键点
        embedding = face.normed_embedding  # 特征向量
        
        # 在注册库中搜索最匹配的身份
        best_match_id = "unknown"
        best_similarity = 0
        
        for known_id, known_embedding in known_faces_db.items():
            # 计算余弦相似度 (向量点积，因为已经归一化)
            similarity = np.dot(embedding, known_embedding)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match_id = known_id
        
        # 应用阈值判断
        if best_similarity < threshold:
            best_match_id = "unknown"
        
        # 构建结果
        result = {
            'face_id': i,
            'bbox': bbox.tolist(),
            'confidence': float(confidence),
            'landmarks': landmarks.tolist(),
            'identity': best_match_id,
            'similarity': float(best_similarity),
            'embedding': embedding  # 可选：如果需要保存原始特征向量
        }
        
        results.append(result)
    
    return results

# #########################
# 4. 可视化函数（可选）
# #########################

def draw_recognition_results(image_path, results, output_path=None):
    """
    在图像上绘制识别结果
    """
    img = cv2.imread(image_path)
    if img is None:
        return
    
    for result in results:
        bbox = result['bbox']
        identity = result['identity']
        similarity = result['similarity']
        
        # 绘制边界框
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        
        # 绘制标签
        label = f"{identity}: {similarity:.3f}"
        cv2.putText(img, label, (bbox[0], bbox[1]-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    if output_path:
        cv2.imwrite(output_path, img)
    
    # 显示图像（可选）
    cv2.imshow('Recognition Results', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# #########################
# 5. 使用示例
# #########################

if __name__ == "__main__":
    # 设置路径
    database_dir = "./database"  # 已知人脸图像文件夹
    test_image_path = "./test.jpg"  # 待测试图像
    
    # 步骤1: 构建注册库
    print("正在构建人脸注册库...")
    build_face_database(database_dir)
    
    if not known_faces_db:
        print("注册库为空，请检查数据库路径和图像内容")
        exit()
    
    # 步骤2: 进行人脸识别
    print(f"\n正在识别图像: {test_image_path}")
    recognition_results = recognize_face(test_image_path, threshold=0.6)
    
    # 步骤3: 输出结果
    print("\n识别结果:")
    for result in recognition_results:
        print(f"人脸 {result['face_id']}:")
        print(f"  位置: {result['bbox']}")
        print(f"  置信度: {result['confidence']:.3f}")
        print(f"  身份: {result['identity']}")
        print(f"  相似度: {result['similarity']:.3f}")
        print("-" * 40)
    
    # 步骤4: 可视化结果（可选）
    print("\n生成可视化结果...")
    draw_recognition_results(test_image_path, recognition_results, "result_output.jpg")