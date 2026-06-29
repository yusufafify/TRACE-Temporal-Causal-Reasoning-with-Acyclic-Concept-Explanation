import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

class Autoencoder(nn.Module):
    def __init__(self, input_shape, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_shape, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.LeakyReLU(0.1)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(0.1),
            nn.Linear(latent_dim, input_shape),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

class AutoencoderTrainer:
    def __init__(self, 
                 autoencoder_cfg,
                 input_shape, 
                 device='cpu'):
        
        self.latend_dim = autoencoder_cfg.latent_dim
        self.noise_level = autoencoder_cfg.noise
        self.lr = autoencoder_cfg.lr
        self.epochs = autoencoder_cfg.epochs
        self.batch_size = autoencoder_cfg.batch_size
        self.patience = autoencoder_cfg.patience

        self.model = Autoencoder(input_shape, self.latend_dim)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.device = device

    def train(self, 
              dataset, 
              selected_var_index=[]):

        train_data = dataset.data['train'].complete_c[:, selected_var_index]
        val_data = dataset.data['val'].complete_c[:, selected_var_index]
        test_data = dataset.data['test'].complete_c[:, selected_var_index]

        n_train = train_data.shape[0]
        n_val = val_data.shape[0]
        n_test = test_data.shape[0]

        # Concatenate over the first dimension
        concat_data = torch.cat([train_data, val_data, test_data], dim=0)

        # # Normalize the data
        # mean = concat_data.mean(dim=0, keepdim=True)
        # std = concat_data.std(dim=0, keepdim=True)
        # concat_data = (concat_data - mean) / std

        data_loader = DataLoader(concat_data, batch_size=self.batch_size)

        if 'gpu' in self.device:
            self.model.to(self.device)

        best_loss = float('inf')
        patience_counter = 0

        print('Autoencoder training started...')
        for epoch in tqdm(range(self.epochs)):
            self.model.train()
            train_loss = 0.0
            for data in data_loader:
                if 'gpu' in self.device:
                    data = data.to(self.device)
                self.optimizer.zero_grad()
                _, outputs = self.model(data)
                loss = self.criterion(outputs, data)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            if epoch % 300 == 0:
                print(f'Epoch {epoch+1}/{self.epochs}, Train Loss: {train_loss:.4f}')

            if train_loss < best_loss:
                best_loss = train_loss
                patience_counter = 0
                best_model_wts = self.model.state_dict()
            else:
                patience_counter += 1
                
            if patience_counter >= self.patience:
                print('Early stopping')
                break
        
        print(f'Epoch {epoch+1}/{self.epochs}, Final Train Loss: {train_loss:.4f}')
        self.model.load_state_dict(best_model_wts)

        # Generate the latent representations
        self.model.eval()
        latent_representations = []
        with torch.no_grad():
            for data in data_loader:
                if 'gpu' in self.device:
                    data = data.to(self.device)
                encoded, _ = self.model(data)
                if self.noise_level > 0:
                    encoded = (1 - self.noise_level)*encoded + self.noise_level*torch.randn_like(encoded)
                latent_representations.append(encoded)

        latent_representations = torch.cat(latent_representations, dim=0)

        dataset.data['train'].X = latent_representations[:n_train, :]
        dataset.data['val'].X = latent_representations[n_train:n_train+n_val, :]
        dataset.data['test'].X = latent_representations[n_train+n_val:, :]
        return dataset


def scale_embeddings(dataset):
    train_data = dataset.data['train'].X
    val_data = dataset.data['val'].X
    test_data = dataset.data['test'].X

    # scaler
    scaler = StandardScaler()
    scaler.fit(train_data)
    dataset.data['train'].X = torch.Tensor(scaler.transform(train_data))
    dataset.data['val'].X = torch.Tensor(scaler.transform(val_data))
    dataset.data['test'].X = torch.Tensor(scaler.transform(test_data))

    return dataset
