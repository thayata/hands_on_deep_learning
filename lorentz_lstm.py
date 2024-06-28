import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Generate Lorenz time series data
def lorenz(t, state, sigma=10.0, rho=28.0, beta=8.0/3.0):
    x, y, z = state
    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z
    return [dx_dt, dy_dt, dz_dt]

t_span = (0, 50)
t_eval = np.linspace(0, 50, 10000)
initial_state = [1.0, 1.0, 1.0]

sol = solve_ivp(lorenz, t_span, initial_state, t_eval=t_eval)
data = sol.y.T

# Preprocessing
scaler = MinMaxScaler(feature_range=(-1, 1))
data_normalized = scaler.fit_transform(data)

def create_sequences(data, seq_length):
    xs = []
    ys = []
    for i in range(len(data) - seq_length):
        x = data[i:i + seq_length]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

seq_length = 10
X, y = create_sequences(data_normalized, seq_length)

# Function to select training interval
def select_training_interval(X, y, start, end):
    return X[start:end], y[start:end]

# Choose start and end points of the training interval
start, end = 0, 2000  # Modify these values as needed

X_train, y_train = select_training_interval(X, y, start, end)
X_test, y_test = X[end:], y[end:]

# Convert data to tensors
X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).float()
X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test).float()

class LSTM(nn.Module):
    def __init__(self, input_size=3, hidden_layer_size=100, output_size=3):
        super(LSTM, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size),
                            torch.zeros(1, 1, self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

model = LSTM()
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 100
for i in range(epochs):
    for seq, labels in zip(X_train, y_train):
        optimizer.zero_grad()
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                             torch.zeros(1, 1, model.hidden_layer_size))

        y_pred = model(seq)

        single_loss = loss_function(y_pred, labels)
        single_loss.backward()
        optimizer.step()

    if i % 10 == 0:
        print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')

# Make predictions
model.eval()
test_predictions = []

for seq in X_test:
    with torch.no_grad():
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                             torch.zeros(1, 1, model.hidden_layer_size))
        test_predictions.append(model(seq).cpu().numpy())

test_predictions = scaler.inverse_transform(np.array(test_predictions).reshape(-1, 3))
y_test = scaler.inverse_transform(y_test.cpu().numpy().reshape(-1, 3))

plt.figure(figsize=(10, 6))
plt.plot(t_eval, data, label=['True X', 'True Y', 'True Z'])
#plt.plot(t_eval[start:end], scaler.inverse_transform(data_normalized)[start:end], label=['Used for learning X', 'Used for learning Y', 'Used for learning Z'])
plt.plot(t_eval[len(data_normalized) - len(test_predictions):len(data_normalized)], test_predictions, label=['Predicted X', 'Predicted Y', 'Predicted Z'])
plt.legend()
plt.show()
