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
    
    import cv2
import numpy as np
import onnxruntime as ort
from insightface.utils import face_align

class FaceRecognizer:
    def __init__(self, model_path, ctx_id=0):
        # 直接加载ONNX模型
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if ctx_id >= 0 else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
    def recognize(self, img):
        # 预处理图像
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))
        img = img.astype(np.float32)
        img = (img - 127.5) / 128.0
        img = np.expand_dims(img, axis=0)
        
        # 运行推理
        embedding = self.session.run([self.output_name], {self.input_name: img})[0]
        
        # 归一化特征向量
        embedding = embedding / np.linalg.norm(embedding)
        
        return [type('obj', (object,), {
            'normed_embedding': embedding[0],
            'embedding': embedding[0]
        })]

if __name__ == "__main__":
    model_path = './w600k_r50.onnx'  # 替换为您的模型路径
    
    recognizer = FaceRecognizer(model_path=model_path, ctx_id=0)
    
    img_path = './test_image.png' 
    img = cv2.imread(img_path)

    if img is None:
        print(f"无法读取图片: {img_path}")
        exit()

    # 确保图像尺寸为112x112
    if img.shape[:2] != (112, 112):
        print(f"调整图像尺寸从 {img.shape[:2]} 到 112x112")
        img = cv2.resize(img, (112, 112))
    
    faces = recognizer.recognize(img)

    if not faces:
        print("特征提取失败。")
    else:
        # 只输出人脸特征向量
        embedding = faces[0].normed_embedding
        print(list(embedding))

    