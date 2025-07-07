import math
import tensorflow as tf
from tensorflow import Tensor
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input,
                                     Conv2D,
                                     Conv2DTranspose,
                                     ReLU,
                                     BatchNormalization,
                                     Add,
                                     Cropping2D)

# Inicialización de pesos con distribución RandomNormal
initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=42)

# Capa para extraer parches de la imagen
class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="SAME",
        )
        # Cada parche se aplana a un vector
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

# Capa para codificar los parches, añadiendo una proyección lineal y embedding posicional
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim, kernel_initializer=initializer)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patches):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patches) + self.position_embedding(positions)
        return encoded

# Bloque Transformer (atención multi-cabeza + FFN)
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [layers.Dense(ff_dim, activation="relu", kernel_initializer=initializer),
             layers.Dense(embed_dim, kernel_initializer=initializer)]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=None):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# Función de activación ReLU seguida de BatchNormalization
def relu_bn(inputs: Tensor) -> Tensor:
    relu = ReLU()(inputs)
    bn = BatchNormalization()(relu)
    return bn

# Bloque residual similar al de ResNet
def residual_block(x: Tensor, downsample: bool, filters: int, kernel_size: int = 3) -> Tensor:
    y = Conv2D(filters=filters,
               kernel_size=kernel_size,
               strides=(1 if not downsample else 2),
               padding="same",
               kernel_initializer=initializer)(x)
    y = relu_bn(y)
    y = Conv2D(filters=filters,
               kernel_size=kernel_size,
               strides=1,
               padding="same",
               kernel_initializer=initializer)(y)

    if downsample:
        x = Conv2D(filters=filters,
                   kernel_size=1,
                   strides=2,
                   padding="same",
                   kernel_initializer=initializer)(x)
    out = Add()([x, y])
    out = relu_bn(out)
    return out

# Función para construir el generador
def Generator(input_shape=(413, 1024, 1),
              patch_size=16,
              projection_dim=256,
              num_heads=4,
              ff_dim=512):
    # Cálculo de dimensiones de la rejilla de parches
    input_height, input_width, channels = input_shape
    # Ahora: 413/16 ≈ 25.81 -> ceil = 26 parches en altura
    num_patches_h = math.ceil(input_height / patch_size)   # 413/16 = 26
    #       1024/16 = 64 parches en anchura
    num_patches_w = math.ceil(input_width / patch_size)      # 1024/16 = 64
    total_patches = num_patches_h * num_patches_w             # 26 * 64 = 1664

    inputs = Input(shape=input_shape)

    # Extracción y codificación de parches
    patches = Patches(patch_size)(inputs)
    encoded_patches = PatchEncoder(total_patches, projection_dim)(patches)

    # Una secuencia de bloques transformer
    x = TransformerBlock(projection_dim, num_heads, ff_dim)(encoded_patches)
    x = TransformerBlock(projection_dim, num_heads, ff_dim)(x)
    x = TransformerBlock(projection_dim, num_heads, ff_dim)(x)
    x = TransformerBlock(projection_dim, num_heads, ff_dim)(x)

    # Reorganizamos la secuencia en un mapa 2D.
    # El reshape pasa de (batch, total_patches, projection_dim) a (batch, num_patches_h, num_patches_w, projection_dim)
    x = layers.Reshape((num_patches_h, num_patches_w, projection_dim))(x)

    # Ahora, realizamos upsampling para llegar a la dimensión deseada.
    # Nuestro mapa actual es de 26x64; queremos llegar a 416x1024 (luego recortamos para obtener 413 en altura).
    # Notamos que 26*16 = 416 y 64*16 = 1024, es decir, un factor de 16 en cada dimensión.
    # Podemos conseguirlo aplicando 4 veces un upsampling por factor 2 (2^4=16).

    filters = 256
    for i in range(4):
        x = Conv2DTranspose(filters, kernel_size=5, strides=2, padding='same',
                             use_bias=False, kernel_initializer=initializer)(x)
        x = BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        # Bloque residual opcional
        x = residual_block(x, downsample=False, filters=filters)
        # Disminuir el número de filtros progresivamente
        filters = max(filters // 2, 32)

    # Última capa para obtener 1 canal con activación tanh
    x = Conv2D(filters=1, kernel_size=3, strides=1, padding='same',
               use_bias=False, activation='tanh', kernel_initializer=initializer)(x)

    # Debido a que el alto resultante es 416 (26*16) y necesitamos 413, recortamos 3 píxeles en el alto.
    x = Cropping2D(cropping=((0, 3), (0, 0)))(x)

    model = Model(inputs=inputs, outputs=x)
    return model

# Ejemplo de creación del modelo
# if __name__ == '__main__':
#     gen = Generator()
#     gen.summary()
