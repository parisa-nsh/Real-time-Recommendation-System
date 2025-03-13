import tensorflow as tf
import numpy as np
from typing import List, Tuple, Dict
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RecommendationModel(tf.keras.Model):
    """Neural Collaborative Filtering model for recommendations."""
    
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 50,
        dense_layers: List[int] = [256, 128, 64],
        dropout_rate: float = 0.3
    ):
        """
        Initialize the recommendation model.
        
        Args:
            num_users: Total number of users in the system
            num_items: Total number of items in the system
            embedding_dim: Dimension of embedding vectors
            dense_layers: List of units in dense layers
            dropout_rate: Dropout rate for regularization
        """
        super(RecommendationModel, self).__init__()
        
        # User and item embedding layers
        self.user_embedding = tf.keras.layers.Embedding(
            num_users,
            embedding_dim,
            embeddings_initializer="he_normal",
            embeddings_regularizer=tf.keras.regularizers.l2(1e-6)
        )
        
        self.item_embedding = tf.keras.layers.Embedding(
            num_items,
            embedding_dim,
            embeddings_initializer="he_normal",
            embeddings_regularizer=tf.keras.regularizers.l2(1e-6)
        )
        
        # Neural network layers
        self.dense_layers = []
        for units in dense_layers:
            self.dense_layers.extend([
                tf.keras.layers.Dense(units, activation='relu'),
                tf.keras.layers.Dropout(dropout_rate),
                tf.keras.layers.BatchNormalization()
            ])
            
        self.final_dense = tf.keras.layers.Dense(1, activation='sigmoid')
        
    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor], training: bool = False) -> tf.Tensor:
        """
        Forward pass of the model.
        
        Args:
            inputs: Tuple of (user_ids, item_ids)
            training: Whether in training mode
            
        Returns:
            Predicted ratings/scores
        """
        user_ids, item_ids = inputs
        
        # Get embeddings
        user_embedding = self.user_embedding(user_ids)
        item_embedding = self.item_embedding(item_ids)
        
        # Concatenate embeddings
        x = tf.concat([user_embedding, item_embedding], axis=1)
        
        # Pass through dense layers
        for layer in self.dense_layers:
            x = layer(x, training=training)
            
        return self.final_dense(x)
    
    def train_step(self, data: Tuple[Tuple[tf.Tensor, tf.Tensor], tf.Tensor]) -> Dict[str, float]:
        """
        Training step logic.
        
        Args:
            data: Tuple of ((user_ids, item_ids), labels)
            
        Returns:
            Dict of metrics
        """
        (user_ids, item_ids), labels = data
        
        with tf.GradientTape() as tape:
            predictions = self((user_ids, item_ids), training=True)
            loss = self.compiled_loss(labels, predictions)
            
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        self.compiled_metrics.update_state(labels, predictions)
        metrics = {m.name: m.result() for m in self.metrics}
        metrics['loss'] = loss
        
        return metrics
    
    def get_top_k_recommendations(
        self,
        user_id: int,
        item_pool: List[int],
        k: int = 10
    ) -> List[Tuple[int, float]]:
        """
        Get top-k recommendations for a user.
        
        Args:
            user_id: User ID to get recommendations for
            item_pool: List of candidate items
            k: Number of recommendations to return
            
        Returns:
            List of (item_id, score) tuples
        """
        # Create input tensors
        user_ids = tf.constant([user_id] * len(item_pool))
        item_ids = tf.constant(item_pool)
        
        # Get predictions
        scores = self.predict((user_ids, item_ids))
        
        # Get top-k items
        top_k_indices = tf.argsort(scores.flatten(), direction='DESCENDING')[:k]
        top_k_items = [item_pool[i] for i in top_k_indices]
        top_k_scores = scores.flatten()[top_k_indices]
        
        return list(zip(top_k_items, top_k_scores.numpy())) 