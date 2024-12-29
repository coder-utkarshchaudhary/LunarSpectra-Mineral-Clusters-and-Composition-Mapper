import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, losses

ARRAY_PATH = r"pixel_vector_data_train_files_transformed.npy"
NUM_CLUSTERS = 100 # To be decided based on number of minerals in the look up table
VECTOR_END=115
VECTOR_START=7

class VAE(tf.keras.Model):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = models.Sequential([
            layers.InputLayer(input_shape=(input_dim,)),
            layers.Dense(hidden_dim*2, activation=tf.nn.gelu),
            layers.Dense(hidden_dim, activation=tf.nn.tanh),
            layers.Dense(hidden_dim//2, activation=tf.nn.gelu),
            layers.Dense(2 * latent_dim)  # Output both mean and log variance
        ])
        
        # Decoder
        self.decoder = models.Sequential([
            layers.InputLayer(input_shape=(latent_dim,)),
            layers.Dense(hidden_dim//2, activation=tf.nn.gelu),
            layers.Dense(hidden_dim, activation=tf.nn.tanh),
            layers.Dense(hidden_dim*2, activation=tf.nn.gelu),
            layers.Dense(input_dim, activation=tf.nn.sigmoid)
        ])
    
    def encode(self, x):
        h = self.encoder(x)
        # mu, var are mean and variance respectively
        mu, logvar = tf.split(h, num_or_size_splits=2, axis=1)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = tf.exp(0.5 * logvar)
        eps = tf.random.normal(shape=tf.shape(std))
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def call(self, inputs):
        mu, logvar = self.encode(inputs)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)
        return reconstructed, mu, logvar, z
    
class GMM(tf.keras.layers.Layer):
    def __init__(self, n_clusters, latent_dim):
        super(GMM, self).__init__()
        self.n_clusters = n_clusters
        self.latent_dim = latent_dim
        
        # GMM parameters
        self.pi = self.add_weight(shape=(n_clusters,), initializer='uniform', trainable=True)
        self.mu_c = self.add_weight(shape=(n_clusters, latent_dim), initializer='random_normal', trainable=True)
        self.logvar_c = self.add_weight(shape=(n_clusters, latent_dim), initializer='random_normal', trainable=True)
    
    def call(self, z):
        z_expand = tf.expand_dims(z, 1)
        mu_expand = tf.expand_dims(self.mu_c, 0)
        logvar_expand = tf.expand_dims(self.logvar_c, 0)
        
        log_probs = -0.5 * (logvar_expand + (z_expand - mu_expand) ** 2 / tf.exp(logvar_expand))
        log_probs = tf.reduce_sum(log_probs, axis=2)
        log_probs = log_probs + tf.math.log(self.pi)
        log_probs = tf.nn.log_softmax(log_probs, axis=1)
        
        return log_probs
    
class VaDE(tf.keras.Model):
    def __init__(self, input_dim, hidden_dim, latent_dim, n_clusters):
        super(VaDE, self).__init__()
        self.vae = VAE(input_dim, hidden_dim, latent_dim)
        self.gmm = GMM(n_clusters, latent_dim)
    
    def call(self, inputs):
        reconstructed, mu, logvar, z = self.vae(inputs)
        log_probs = self.gmm(z)
        return reconstructed, mu, logvar, z, log_probs
    
    def compute_loss(self, x, reconstructed, mu, logvar, z, log_probs):
        recon_loss = tf.reduce_sum(losses.binary_crossentropy(x, reconstructed))
        kl_div = -0.5 * tf.reduce_sum(1 + logvar - tf.square(mu) - tf.exp(logvar))
        kl_gmm = -tf.reduce_sum(tf.reduce_sum(log_probs * tf.exp(log_probs), axis=1))
        
        return recon_loss + kl_div + kl_gmm


def train_vade(model, data, epochs, batch_size, learning_rate):
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    
    dataset = tf.data.Dataset.from_tensor_slices(data).shuffle(buffer_size=1024).batch(batch_size)
    
    for epoch in range(epochs):
        epoch_loss = 0
        for batch in dataset:
            with tf.GradientTape() as tape:
                reconstructed, mu, logvar, z, log_probs = model(batch)
                loss = model.compute_loss(batch, reconstructed, mu, logvar, z, log_probs)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            epoch_loss += loss.numpy()
        
        print(f'Epoch {epoch+1}, Loss: {epoch_loss / len(data):.4f}')

def cluster_data(model, data):
    _, _, _, _, log_probs = model(data)
    clusters = tf.argmax(log_probs, axis=1)
    return clusters.numpy()

if __name__ == "__main__":
    input_dimension = VECTOR_END-VECTOR_START
    hidden_dimension = 128
    latent_dimension = 32
    n_clusters = NUM_CLUSTERS

    X = np.load(ARRAY_PATH)
    X = X[:12851*250]
    X = X.reshape(-1, input_dimension)
    
    print(f"Shape of X is {X.shape}")

    vade_model = VaDE(input_dimension, hidden_dimension, latent_dimension, n_clusters)
    train_vade(vade_model, X, epochs=20, batch_size=128, learning_rate=3e-3)

    vade_model.save("vade_model.h5")
    
    # Cluster wala function will be used here to extract the cluster number assignment for a particular sample.
    for i in range(X.shape[0]):
        clusters = cluster_data(vade_model, X)

    print(clusters.shape)
    