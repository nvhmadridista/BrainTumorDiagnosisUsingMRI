
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.applications import ConvNeXtTiny

def build_hybrid_model(input_shape=(224, 224, 3), num_classes=4):
    """
    Định nghĩa kiến trúc Hybrid: ConvNeXt-Tiny + Transformer Encoder
    """
    inputs = Input(shape=input_shape)
    
    # 1. Backbone: ConvNeXtTiny (Feature Extractor)
    # include_top=False để bỏ lớp phân loại gốc
    backbone = ConvNeXtTiny(include_top=False, weights='imagenet', input_tensor=inputs)
    
    # Mặc định đóng băng backbone (người train có thể mở sau)
    backbone.trainable = False 
    
    x = backbone.output 
    
    # 2. Flatten -> Tokens (Chuẩn bị cho Transformer)
    # Output của ConvNeXt là (7, 7, 768) -> Reshape thành (49, 768)
    x = layers.Reshape((-1, x.shape[-1]))(x) 
    
    # 3. Transformer Encoder Block
    embed_dim = 768
    num_heads = 8
    ff_dim = 1024
    
    # Layer Norm 1 + Attention
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(x, x)
    x = layers.Add()([x, attention_output])
    
    # Layer Norm 2 + Feed Forward
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    ffn = models.Sequential([
        layers.Dense(ff_dim, activation="relu"),
        layers.Dense(embed_dim),
    ])
    ffn_output = ffn(x)
    x = layers.Add()([x, ffn_output])
    
    # 4. Classifier Head
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name="ConvNeXt_Transformer_Hybrid")
    return model

if __name__ == "__main__":
    # Code test nhanh để kiểm tra kiến trúc khi chạy trực tiếp file này
    model = build_hybrid_model()
    model.summary()
    print("✅ Model architecture created successfully!")
