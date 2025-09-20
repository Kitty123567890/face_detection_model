import insightface
from insightface.app import FaceAnalysis
import cv2
import numpy as np

class FaceRecognizer:
    def __init__(self, model_name, det_size=(640, 640), ctx_id=0):
        self.model_name = model_name
        self.det_size = det_size
        self.ctx_id = ctx_id
        self.app = self.load_model()


    def load_model(self):
        try:
            app = FaceAnalysis(name=self.model_name, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            app.prepare(ctx_id=self.ctx_id, det_size=self.det_size)
            print("模型加载成功:", self.model_name)
            return app
        except Exception as e:
            print(f"模型加载失败，请检查环境或模型名称: {e}")
            exit()

    def recognize(self, image):
        faces = self.app.get(image)
        return faces
    

if __name__ == "__main__":
    model_name = 'buffalo_l'  # 选择模型名称
    recognizer = FaceRecognizer(model_name=model_name, ctx_id=0, det_size=(640, 640))
    
    img_path = './test_image.jpg' # 替换成你的图片路径
    img = cv2.imread(img_path)

    if img is None:
        print(f"无法读取图片: {img_path}")
        exit()

    faces = recognizer.recognize(img)

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
            
            # 其他信息
            if face.kps is not None:
                print(f"  - 关键点 (Keypoints) 数量: {len(face.kps)}")

    