import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, TensorDataset
from scipy import fft
from scipy.ndimage import uniform_filter1d

if torch.cuda.is_available():
    dev = "cuda"
else:
    dev = "cpu"

print('Using: {}'.format(dev))


class Autoencoder(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = torch.nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=16, stride=8),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            #nn.MaxPool1d(strides=2),
            nn.Conv1d(64, 64, kernel_size=8, stride=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            #nn.MaxPooling1D(strides=2),
            nn.Conv1d(64, 64, kernel_size=4, stride=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            #nn.MaxPooling1D(strides=2),
            nn.Conv1d(64, 64, kernel_size=4, stride=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            #nn.MaxPooling1D(strides=2),
            nn.Conv1d(64, 64, kernel_size=4, stride=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            #nn.Conv1d(64, 64, kernel_size=2, stride=1),
            #nn.BatchNorm1d(64),
            #nn.ReLU(),
            # nn.Conv1d(64, 1, kernel_size=4, stride=2),
            #nn.BatchNorm1d(64),
            #nn.ReLU(),
            nn.Flatten(),
            #nn.Dropout(0.5),
            #nn.Linear(384, 20),
            nn.Linear(128, 20),
            nn.Flatten(),
        )

        self.decoder = torch.nn.Sequential(
            nn.Flatten(),
            nn.Linear(20, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1200),
        )

        '''
        self.decoder = torch.nn.Sequential(
            nn.ConvTranspose1d(1, 32, kernel_size=4, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 32, kernel_size=4, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 32, kernel_size=8, stride=4, padding=1, output_padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 32, kernel_size=8, stride=4, padding=1, output_padding=1),
            nn.BatchNorm1d(32),
            nn.ConvTranspose1d(32, 1, kernel_size=3, stride=1), #, padding=1, output_padding=1),
            #nn.ConvTranspose1d(32, 32, kernel_size=8, stride=4, padding=1, output_padding=1),
            #nn.BatchNorm1d(32),
            nn.Flatten(),
            #nn.Linear(5344, 300),
        )
        '''

    def forward(self, x):
        encoded = self.encoder(x.unsqueeze(1))
        decoded = self.decoder(encoded.unsqueeze(1))
        return decoded
        #return decoded[:, :x.size(1)]

    def encode(self, x):
        return self.encoder(x.unsqueeze(1))


def add_white_noise(x, snr):
    snr_linear = 10**(snr/10)
    noise_power = torch.var(x, dim=1) / snr_linear  # calculate noise power
    stds = torch.sqrt(noise_power).repeat(x.size(1), 1).T.to(dev)
    means = torch.zeros(x.size()).to(dev)
    noise = torch.normal(mean=means, std=stds)  # generate white noise

    return x + noise


def shift(x, y):
    sh = random.randint(0, x.size(1))

    x = torch.roll(x, sh, dims=1)
    y = torch.roll(y, sh, dims=1)

    return x, y


def train(ae, X_train, Y_train, X_test, Y_test):
    X, Y = torch.Tensor(X_train), torch.tensor(Y_train)
    Xt, Yt = torch.Tensor(X_test), torch.tensor(Y_test)

    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(ae.parameters(), lr=0.0001)

    train_dataset = TensorDataset(X, Y)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

    test_dataset = TensorDataset(Xt, Yt)
    test_loader = DataLoader(test_dataset, batch_size=8)

    #test_dataset = TensorDataset(Xt, Yt)
    #test_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

    num_epochs = 100
    for epoch in range(num_epochs):
        losses = []
        for data in train_loader:
            x, y = data
            x = x.to(dev)
            y = y.to(dev)

            '''
            with torch.no_grad():
                x, y = augment(x, snr=30)
            '''

            #x = add_white_noise(x, snr=50)
            x, y = shift(x, y)

            optimizer.zero_grad()
            y_hat = ae(x)

            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()

            losses.append(loss.cpu().item())

        print('Epoch [{}/{}], Loss: {:.8f}'.format(epoch+1, num_epochs, np.mean(losses)))

        if epoch%10 == 0:
            with torch.no_grad():
                #x, y = train_dataset[0:1]
                #x = x.to(dev)
                #y_hat = ae(x)

                xt, yt = test_dataset[0:10]
                xt = xt.to(dev)
                yt_hat = ae(xt)

                # Display original and reconstructed images
                fig, ax = plt.subplots(10, 1, figsize=(15,10))
                for j in range(0, 5):
                    ax[j].plot(y[j].cpu().numpy(), label='given')
                    ax[j].plot(y_hat[j].flatten().cpu().numpy(), label='predicted')
                    ax[j+5].plot(yt[j], label='gt')
                    ax[j+5].plot(yt_hat[j].flatten().cpu().numpy(), label='predicted')
                plt.legend()
                plt.savefig('img/reconstruct.png')
                torch.save(ae.cpu().state_dict(), 'ae_dft.pth')
                ae.to(dev)


        if epoch % 5 == 0:
            with torch.no_grad():
                test_losses = []
                for data in test_loader:
                    x, y = data
                    x = x.to(dev)
                    y = y.to(dev)

                    y_hat = ae(x)

                    loss = criterion(y_hat, y)
                    test_losses.append(loss.item())
                print('* Epoch [{}/{}], Test loss: {:.4f}'.format(epoch+1, num_epochs, np.mean(test_losses)))


    return ae


def preprocess(X):
    X_fft = np.abs(fft.rfft(X[:, :2400]))[:,1:]
    X_fft /= np.max(X_fft, axis=1, keepdims=1)

    Y = uniform_filter1d(X_fft, size=10)
    Y = Y/np.max(Y, axis=1, keepdims=True)

    return Y, Y


if __name__ == '__main__':
    npz = np.load("C:\\Users\\Семья\\Documents\\vibro_analitics\\datasets\\acceleration.npz")
    X = npz['X']
    ind = npz['ind']
    X, Y = preprocess(X)
    X_train = X[ind==0]
    X_test = X[ind==1]
    Y_train = Y[ind==0]
    Y_test = Y[ind==1]
    model = Autoencoder()
#        model.load_state_dict(torch.load(args.checkpoint))
    model.to(dev)
    model = train(
            model,
            X_train.astype(np.float32),
            Y_train.astype(np.float32),
            X_test.astype(np.float32),
            Y_test.astype(np.float32))
    model.to('cpu')
    torch.save(model.state_dict(), './models/ae_dft.pth')
