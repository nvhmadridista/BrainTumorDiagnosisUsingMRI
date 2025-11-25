import onnxruntime as ort
import numpy as np
import os

class ONNXModel:
    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Không tìm thấy file Model ONNX: {model_path}")
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict(self, input_tensor: np.ndarray):
        """
        input_tensor: numpy array, shape (1, 224, 224, 3), dtype float32
        Trả về: (predicted_class:int, softmax_probabilities:np.ndarray)
        """
        # Chạy inference
        outputs = self.session.run([self.output_name], {self.input_name: input_tensor.astype(np.float32)})

        logits = outputs[0]  # shape (1, num_classes)
        # Tính softmax thủ công
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        pred_class = int(np.argmax(probs, axis=1)[0])

        return pred_class, probs[0]
