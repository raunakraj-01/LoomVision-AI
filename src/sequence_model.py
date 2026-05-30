"""
sequence_model.py
-----------------
Unsupervised Predictive LSTM for detecting temporal anomalies in fabric patterns.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque

class LSTMFeaturePredictor(nn.Module):
    def __init__(self, input_dim=384, hidden_dim=128, num_layers=1):
        super(LSTMFeaturePredictor, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        # x is of shape (batch, seq_len, input_dim)
        out, _ = self.lstm(x)
        # We want to predict the next feature vector based on the last output of the LSTM
        out = self.fc(out[:, -1, :])
        return out


class TemporalAnomalyDetector:
    def __init__(self, input_dim=384, seq_len=10, hidden_dim=128, learning_rate=0.001):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LSTMFeaturePredictor(input_dim, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        self.seq_len = seq_len
        self.feature_buffer = deque(maxlen=seq_len + 1) # Needs seq_len context + 1 target
        
        # Track training progress and establish baseline loss
        self.loss_history = deque(maxlen=100)
        self.is_warmed_up = False
        self.warmup_frames = 50

    def process_frame_embedding(self, embedding_tensor, is_spatial_defect=False):
        """
        Process the new frame embedding.
        If the spatial model (PatchCore) detects a defect, we can choose not to train 
        the LSTM on it, so the LSTM only learns "normal" sequences.
        """
        # Ensure it's on the right device and shape
        emb = embedding_tensor.detach().to(self.device).view(-1)
        self.feature_buffer.append(emb)
        
        temporal_score = 0.0
        
        # We need seq_len + 1 frames to train (seq_len for context, 1 for target)
        if len(self.feature_buffer) < self.seq_len + 1:
            return temporal_score, self.is_warmed_up
            
        # Form sequence and target
        buffer_stack = torch.stack(list(self.feature_buffer))
        seq = buffer_stack[:-1].unsqueeze(0) # Shape: (1, seq_len, input_dim)
        target = buffer_stack[-1].unsqueeze(0) # Shape: (1, input_dim)
        
        # Predict next frame
        self.model.eval()
        with torch.no_grad():
            pred = self.model(seq)
            loss = self.criterion(pred, target).item()
            
        # The anomaly score is the prediction error
        temporal_score = float(loss)
        
        # Online Learning (train the LSTM on this clean frame step)
        # We skip training if the spatial engine says this frame is defective,
        # so we don't accidentally train the LSTM to expect defects.
        if not is_spatial_defect:
            self.model.train()
            self.optimizer.zero_grad()
            train_pred = self.model(seq)
            train_loss = self.criterion(train_pred, target)
            train_loss.backward()
            self.optimizer.step()
            
            # Update baseline
            self.loss_history.append(temporal_score)
            
        if len(self.loss_history) >= self.warmup_frames:
            self.is_warmed_up = True
            
        return temporal_score, self.is_warmed_up

    def get_baseline_threshold(self, sigma=3.0):
        if not self.is_warmed_up or len(self.loss_history) < 10:
            return 1.0 # High threshold until warm
        scores = np.array(self.loss_history)
        return float(np.mean(scores) + sigma * np.std(scores))
