import scipy.io as sio
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
import matplotlib
import math
import pandas as pd
import sys
import DataPreprocess as dp
sys.path.append('./')


# Define LSTM Neural Networks
class LstmRNN(nn.Module):
    """
        Parameters：
        - input_size: feature size
        - hidden_size: number of hidden units
        - output_size: number of output
        - num_layers: layers of LSTM to stack
    """

    def __init__(self, input_size, hidden_size=1, output_size=1, num_layers=1):
        super().__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)  # utilize the LSTM model in torch.nn
        self.linear1 = nn.Linear(hidden_size, output_size) # 全连接层

    def forward(self, _x):
        print(_x.shape)
        x, _ = self.lstm(_x)  # _x is input, size (seq_len, batch, input_size)
        s, b, h = x.shape  # x is output, size (seq_len, batch, hidden_size)
        x = x.view(s * b, h)
        x = self.linear1(x)
        x = x.view(s, b, -1)
        print(x.shape)
        return x

if __name__ == '__main__':
    if (torch.cuda.is_available()):
        device = torch.device("cuda:0")
        print('Training on GPU.')
    else:
        print('No GPU available, training on CPU.')
        
    data_x = np.array(pd.read_csv('Data_x.csv', header=None))
    data_y = np.array(pd.read_csv('Data_y.csv', header=None))
    data_x = data_x.astype('float32') # 转换数据类型
    data_y = data_y.astype('float32')

    data_len = len(data_x)
    t = np.linspace(0, data_len, data_len)

    train_data_ratio = 0.8  # Choose 80% of the data for training
    train_data_len = int(data_len * train_data_ratio)

    train_x = data_x[10:train_data_len]
    train_y = data_y[10:train_data_len]
    t_for_training = t[10:train_data_len]

    INPUT_FEATURES_NUM = 5
    OUTPUT_FEATURES_NUM = 1

    test_x = data_x[train_data_len:]
    test_y = data_y[train_data_len:]
    t_for_testing = t[train_data_len:]

    # ----------------- train -------------------
    train_x_tensor = train_x.reshape(-1, 1, INPUT_FEATURES_NUM)  # set batch size to 1
    train_y_tensor = train_y.reshape(-1, 1, OUTPUT_FEATURES_NUM)  # set batch size to 1

    # transfer data to pytorch tensor
    train_x_tensor = torch.from_numpy(train_x_tensor).to(device)
    train_y_tensor = torch.from_numpy(train_y_tensor).to(device)

    lstm_model = LstmRNN(INPUT_FEATURES_NUM, 20, output_size=OUTPUT_FEATURES_NUM, num_layers=1).to(device)  # 20 hidden units
    print('LSTM model:', lstm_model)
    print('model.parameters:', lstm_model.parameters)
    print(next(lstm_model.parameters()).device) # 输出cuda:0表明在cuda上训练

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=1e-2)

    prev_loss = 1000
    max_epochs = 2000
    for epoch in range(max_epochs):
        output = lstm_model(train_x_tensor)
        loss = criterion(output, train_y_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if loss < prev_loss:
            torch.save(lstm_model.state_dict(), 'lstm_model.pt') # save model parameters to files
            prev_loss = loss

        if loss.item() < 1e-4:
            print('Epoch [{}/{}], Loss: {:.5f}'.format(epoch + 1, max_epochs, loss.item()))
            print("The loss value is reached")
            break
        elif (epoch + 1) % 100 == 0:
            print('Epoch: [{}/{}], Loss:{:.5f}'.format(epoch + 1, max_epochs, loss.item()))

    # prediction on training dataset
    pred_y_for_train = lstm_model(train_x_tensor.to(device))
    pred_y_for_train = pred_y_for_train.cpu().view(-1, OUTPUT_FEATURES_NUM).data.numpy()

    pred_y_for_train = dp.Restore(pred_y_for_train, 1, 0)

    # ----------------- test -------------------
    lstm_model = lstm_model.eval()  # switch to testing model

    # prediction on test dataset
    test_x_tensor = test_x.reshape(-1, 1,
                                   INPUT_FEATURES_NUM)  # set batch size to 1, the same value with the training set
    test_x_tensor = torch.from_numpy(test_x_tensor) # 变为tensor

    pred_y_for_test = lstm_model(test_x_tensor.to(device))
    pred_y_for_test = pred_y_for_test.cpu().view(-1, OUTPUT_FEATURES_NUM).data.numpy()

    pred_y_for_test = dp.Restore(pred_y_for_test, 1, 0)
    test_x_tensor = dp.Restore(test_x_tensor, 1, 0)

    train_y = dp.Restore(train_y, 1, 0)
    test_y = dp.Restore(test_y, 1, 0)

    loss = criterion(torch.from_numpy(pred_y_for_test), torch.from_numpy(test_y))
    print("test loss：", loss.item())

    # ----------------- plot -------------------
    plt.figure()
    plt.plot(t_for_training, train_y, 'b', label='y_trn')
    plt.plot(t_for_training, pred_y_for_train, 'y--', label='pre_trn')

    plt.plot(t_for_testing, test_y, 'k', label='y_tst')
    plt.plot(t_for_testing, pred_y_for_test, 'm--', label='pre_tst')

    plt.xlabel('t')
    plt.ylabel('Vce')
    plt.show()