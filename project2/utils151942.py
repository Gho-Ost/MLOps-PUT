import os
from dotenv import load_dotenv

from pathlib import Path

import pandas as pd
import numpy as np

import torch
import torch.nn.functional as F

from torch import nn
from torch.utils.data import Dataset, DataLoader

import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

import optuna
from optuna.integration import PyTorchLightningPruningCallback

import wandb

# from sklearn.preprocessing import StandardScaler

from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt


def plot_seasonal_decomposition(df):
    periods = {
        'Daily': 96,
        'Weekly': 96 * 7,
        'Monthly': 96 * 30,
        'Yearly': 96 * 365
    }

    decompositions = {
        name: seasonal_decompose(df['value'], model='additive', period=period, extrapolate_trend='freq')
        for name, period in periods.items()
    }

    plt.rcParams.update({
        'font.size': 8,
        'axes.titlesize': 9,
        'axes.labelsize': 8,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7
    })

    fig, axes = plt.subplots(len(periods), 3, figsize=(20, 12), dpi=120)
    plt.subplots_adjust(hspace=0.5, wspace=0.3)

    for i, (name, result) in enumerate(decompositions.items()):
        components = [result.trend, result.seasonal, result.resid]
        titles = [f"Trend", "Seasonal", "Residual"]

        for j, (comp, title) in enumerate(zip(components, titles)):
            axes[i, j].plot(comp, linewidth=0.8)
            axes[i, j].set_title(title)
            axes[i, j].tick_params(axis='x', rotation=30)
            if j == 0:
                axes[i, j].set_ylabel(name, fontsize=8)
            else:
                axes[i, j].set_ylabel("")

    fig.suptitle("Seasonal Decomposition — Daily, Weekly, Monthly, and Yearly", fontsize=12, y=1.03)
    plt.tight_layout()
    plt.show()

def plot_consumption(df):
    df['hour'] = df.index.hour
    df['weekday'] = df.index.dayofweek   # 0=Monday, 6=Sunday
    df['month'] = df.index.month
    df['year'] = df.index.year

    avg_by_hour = df.groupby('hour')['value'].mean()
    avg_by_day = df.groupby('weekday')['value'].mean()
    avg_by_month = df.groupby('month')['value'].mean()
    avg_by_year = df.groupby('year')['value'].mean()

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Average Energy Consumption Patterns', fontsize=16)

    axs[0, 0].plot(avg_by_hour.index, avg_by_hour.values, marker='o')
    axs[0, 0].set_title('Average Consumption by Hour of Day')
    axs[0, 0].set_xlabel('Hour')
    axs[0, 0].set_ylabel('Average Consumption')
    axs[0, 0].grid(True)

    day_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    axs[0, 1].bar(avg_by_day.index, avg_by_day.values, color='tab:orange')
    axs[0, 1].set_xticks(range(7))
    axs[0, 1].set_xticklabels(day_labels)
    axs[0, 1].set_title('Average Consumption by Day of Week')
    axs[0, 1].set_xlabel('Day')
    axs[0, 1].set_ylabel('Average Consumption')

    axs[1, 0].plot(avg_by_month.index, avg_by_month.values, marker='s', color='tab:green')
    axs[1, 0].set_xticks(range(1, 13))
    axs[1, 0].set_title('Average Consumption by Month')
    axs[1, 0].set_xlabel('Month')
    axs[1, 0].set_ylabel('Average Consumption')
    axs[1, 0].grid(True)

    axs[1, 1].bar(avg_by_year.index, avg_by_year.values, color='tab:purple')
    axs[1, 1].set_title('Average Consumption by Year')
    axs[1, 1].set_xlabel('Year')
    axs[1, 1].set_ylabel('Average Consumption')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def encode_calendar_features(timestamps):
    hours = timestamps.hour + timestamps.minute / 60.0
    hour_sin = np.sin(2 * np.pi * hours / 24)
    hour_cos = np.cos(2 * np.pi * hours / 24)

    dow = timestamps.dayofweek
    dow_sin = np.sin(2 * np.pi * dow / 7)
    dow_cos = np.cos(2 * np.pi * dow / 7)

    doy = timestamps.dayofyear
    doy_sin = np.sin(2 * np.pi * doy / 365)
    doy_cos = np.cos(2 * np.pi * doy / 365)

    is_weekend = (dow >= 5).astype(float)

    features = np.stack(
        [hour_sin, hour_cos, dow_sin, dow_cos, doy_sin, doy_cos, is_weekend],
        axis=1
    )
    return features

class TimeSeriesDataset(Dataset):
    def __init__(self, values, timestamps, seq_length, pred_length, add_time_features=False):
        self.values = values
        self.timestamps = timestamps
        self.seq_length = seq_length
        self.pred_length = pred_length
        self.add_time_features = add_time_features

        if self.add_time_features:
            self.time_features = encode_calendar_features(timestamps)

    def __len__(self):
        return len(self.values) - self.seq_length - self.pred_length + 1

    def __getitem__(self, idx):
        x = self.values[idx : idx + self.seq_length]
        y = self.values[idx + self.seq_length : idx + self.seq_length + self.pred_length]

        if self.add_time_features:
            x_time = self.time_features[idx : idx + self.seq_length]
            x = np.concatenate([x[:, None], x_time], axis=1)
            return torch.FloatTensor(x), torch.FloatTensor(y).unsqueeze(-1)

        return torch.FloatTensor(x).unsqueeze(-1), torch.FloatTensor(y).unsqueeze(-1)


class BaseLSTMForecaster(L.LightningModule):
    def __init__(
        self,
        input_size = 1,
        hidden_size = 64,
        num_layers = 2,
        dropout = 0.2,
        learning_rate = 1e-3,
        pred_length = 1,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.fc = nn.Linear(hidden_size, pred_length)
        self.learning_rate = learning_rate
        
    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)

        last_output = lstm_out[:, -1, :]
        predictions = self.fc(last_output)
        return predictions.unsqueeze(-1)  # (batch, pred_length, 1)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.mse_loss(y_hat, y)
        mae = nn.functional.l1_loss(y_hat, y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_mae', mae, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.mse_loss(y_hat, y)
        mae = nn.functional.l1_loss(y_hat, y)
        self.log('test_loss', loss)
        self.log('test_mae', mae)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }


class TimeSeriesDataModule(L.LightningDataModule):
    def __init__(
        self,
        df: pd.DataFrame,
        value_column: str,
        seq_length=96,
        pred_length=1,
        batch_size=32,
        add_time_features=False):

        super().__init__()
        self.df = df
        self.value_column = value_column
        self.seq_length = seq_length
        self.pred_length = pred_length
        self.batch_size = batch_size
        self.add_time_features = add_time_features

    def setup(self, stage=None):
        self.df = self.df.sort_index()
        data = self.df[self.value_column].values
        timestamps = self.df.index

        test_size = 4 * 24 * 365  # one year of 15-min steps
        train_end = len(data) - test_size - self.pred_length + 1

        train_data = data[:train_end]
        train_ts = timestamps[:train_end]

        test_data = data[train_end - self.seq_length:]
        test_ts = timestamps[train_end - self.seq_length:]

        val_size = int(len(train_data) * 0.1)
        train_data_final = train_data[:-val_size]
        val_data = train_data[-val_size - self.seq_length:]

        train_ts_final = train_ts[:-val_size]
        val_ts = train_ts[-val_size - self.seq_length:]

        # print(f"Data split summary:")
        # print(f"  Total samples: {len(data)}")
        # print(f"  Training samples: {len(train_data_final)}")
        # print(f"  Validation samples: {len(val_data)}")
        # print(f"  Test samples: {len(test_data)}")
        # print(f"  Sequence length: {self.seq_length}")
        # print(f"  Prediction length: {self.pred_length}")

        self.train_dataset = TimeSeriesDataset(train_data_final, train_ts_final, self.seq_length, self.pred_length, self.add_time_features)
        self.val_dataset = TimeSeriesDataset(val_data, val_ts, self.seq_length, self.pred_length, self.add_time_features)
        self.test_dataset = TimeSeriesDataset(test_data, test_ts, self.seq_length, self.pred_length, self.add_time_features)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0, pin_memory=True)


class TimeSeriesExperiment:
    def __init__(
        self,
        df: pd.DataFrame,
        value_column,
        seq_length=96,
        pred_length=1,
        batch_size=32,
        wandb_project="timeseries-lstm",
        wandb_entity=None,
        add_time_features=False):

        self.datamodule = TimeSeriesDataModule(
            df=df,
            value_column=value_column,
            seq_length=seq_length,
            pred_length=pred_length,
            batch_size=batch_size,
            add_time_features=add_time_features
        )
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity

    def train_model(
        self,
        input_size=1,
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
        learning_rate=1e-3,
        max_epochs=100,
        experiment_name=None,
        dev_run_flag=False):

        wandb_logger = WandbLogger(
            project=self.wandb_project,
            entity=self.wandb_entity,
            name=experiment_name or f"pred_{self.datamodule.pred_length}_h{hidden_size}_l{num_layers}",
            log_model=True,
        )

        model = BaseLSTMForecaster(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            learning_rate=learning_rate,
            pred_length=self.datamodule.pred_length,
        )

        early_stop = EarlyStopping(monitor="val_loss", patience=15, mode="min", verbose=True)
        checkpoint = ModelCheckpoint(
            monitor="val_loss", mode="min", save_top_k=1, filename="best-{epoch:02d}-{val_loss:.4f}"
        )

        trainer = L.Trainer(
            max_epochs=max_epochs,
            logger=wandb_logger,
            callbacks=[early_stop, checkpoint],
            accelerator="auto",
            devices=1,
            gradient_clip_val=1.0,
            enable_progress_bar=True,
            num_sanity_val_steps=0,
            fast_dev_run=dev_run_flag,
        )

        trainer.fit(model, datamodule=self.datamodule)
        trainer.test(model, datamodule=self.datamodule, ckpt_path="best", verbose=True)

        wandb.finish()
        return model, trainer

    def optimize_hyperparameters(self, n_trials=10, max_epochs=50, dev_run_flag=False):
        num_layers = 2
        dropout = 0.2

        def objective(trial):
            hidden_size = trial.suggest_int("hidden_size", 8, 32, step=4)
            learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)

            wandb_logger = WandbLogger(
                project=self.wandb_project,
                entity=self.wandb_entity,
                name=f"trial_{trial.number}",
                log_model=False,
            )

            model = BaseLSTMForecaster(
                input_size=1,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                learning_rate=learning_rate,
                pred_length=self.datamodule.pred_length,
            )

            pruning_callback = PyTorchLightningPruningCallback(trial, monitor="val_loss")

            trainer = L.Trainer(
                max_epochs=max_epochs,
                logger=wandb_logger,
                callbacks=[pruning_callback],
                accelerator="auto",
                devices=1,
                enable_progress_bar=False,
                enable_checkpointing=False,
                fast_dev_run=dev_run_flag,
                num_sanity_val_steps=0
            )

            trainer.fit(model, datamodule=self.datamodule)

            wandb.finish()
            return trainer.callback_metrics["val_loss"].item()

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)

        print("\nBest hyperparameters:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")
        print(f"\nBest validation loss: {study.best_value:.4f}")

        return study.best_params

    def plot_predictions(self, model, n_examples=3, omit_time_features=True):
        model.eval()
        test_loader = self.datamodule.test_dataloader()

        example_batches = []

        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(test_loader):
                y_pred = model(x)

                if batch_idx < n_examples:
                    example_batches.append((x.cpu(), y.cpu(), y_pred.cpu()))
                else:
                    break

        for i, (x, y, y_pred) in enumerate(example_batches):
            plt.figure(figsize=(10, 4))
            
            if omit_time_features:
                input_seq = x[0, :, 0].numpy()
            else:
                input_seq = x[0].numpy().mean(axis=-1)

            true_future = y[0].squeeze(-1).numpy()
            pred_future = y_pred[0].squeeze(-1).numpy()

            plt.plot(range(len(input_seq)), input_seq, label='Input Sequence')
            plt.plot(
                range(len(input_seq), len(input_seq) + len(true_future)),
                true_future,
                label='True Future',
            )
            plt.plot(
                range(len(input_seq), len(input_seq) + len(pred_future)),
                pred_future,
                label='Predicted Future',
            )
            plt.title(f"Example Prediction {i+1}")
            plt.xlabel("Time Step")
            plt.ylabel("Value")
            plt.legend()
            plt.grid(True)
            plt.show()

def compute_static_feats(df, value_column):
    values = df[value_column].to_numpy()
    
    ts_mean, ts_std = np.mean(values), np.std(values)
    ts_min, ts_max = np.min(values), np.max(values)
    
    monthly_stats = df[value_column].resample('M').agg(['mean', 'std', 'sum'])
    monthly_mean = monthly_stats['mean'].mean()
    monthly_std = monthly_stats['std'].mean()
    monthly_sum = monthly_stats['sum'].mean()
    
    percentiles = np.percentile(values, [0, 20, 40, 60, 80, 100])
    trend_slope = np.polyfit(np.arange(len(values)), values, 1)[0] if len(values) > 1 else 0.0
    
    static_features = np.array([
        ts_mean, ts_std, ts_min, ts_max,
        monthly_mean, monthly_std, monthly_sum,
        *percentiles,
        trend_slope
    ], dtype=np.float32)
    
    return static_features

class EnhancedTimeSeriesDataset(Dataset):
    def __init__(self, values, timestamps, df, value_column, seq_length, pred_length, 
                 add_time_features=False, add_static_features=False):
        self.values = values
        self.timestamps = timestamps
        self.df = df
        self.value_column = value_column
        self.seq_length = seq_length
        self.pred_length = pred_length
        self.add_time_features = add_time_features
        self.add_static_features = add_static_features
        
        if self.add_time_features:
            self.time_features = encode_calendar_features(timestamps)
        
        if self.add_static_features:
            self.static_feats = compute_static_feats(df, value_column)

    def __len__(self):
        return len(self.values) - self.seq_length - self.pred_length + 1

    def __getitem__(self, idx):
        x = self.values[idx : idx + self.seq_length]
        y = self.values[idx + self.seq_length : idx + self.seq_length + self.pred_length]
        
        if self.add_time_features:
            x_time = self.time_features[idx : idx + self.seq_length]
            x = np.concatenate([x[:, None], x_time], axis=1)
        else:
            x = x[:, None]
        
        if self.add_static_features:
            return (torch.FloatTensor(x), 
                    torch.FloatTensor(self.static_feats),
                    torch.FloatTensor(y).unsqueeze(-1))
        
        return torch.FloatTensor(x), torch.FloatTensor(y).unsqueeze(-1)

    
class StaticDynamicLSTM(L.LightningModule):
    def __init__(
        self,
        dynamic_input_size=1,
        static_input_size=0,
        hidden_size=64,
        static_embedding_size=32,
        num_layers=2,
        dropout=0.2,
        learning_rate=1e-3,
        pred_length=1,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.use_static = static_input_size > 0
        
        self.lstm = nn.LSTM(
            input_size=dynamic_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        if self.use_static:
            self.static_encoder = nn.Sequential(
                nn.Linear(static_input_size, static_embedding_size * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(static_embedding_size * 2, static_embedding_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            
            fusion_input_size = hidden_size + static_embedding_size
        else:
            fusion_input_size = hidden_size
        
        self.output_layers = nn.Sequential(
            nn.Linear(fusion_input_size, fusion_input_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_input_size // 2, pred_length)
        )
        
        self.learning_rate = learning_rate
        
    def forward(self, dynamic_input, static_input=None):
        lstm_out, (h_n, c_n) = self.lstm(dynamic_input)
        dynamic_repr = lstm_out[:, -1, :]  # (batch, hidden_size)
        
        if self.use_static and static_input is not None:
            static_repr = self.static_encoder(static_input)  # (batch, static_embedding_size)
            fused = torch.cat([dynamic_repr, static_repr], dim=1)
        else:
            fused = dynamic_repr
        
        predictions = self.output_layers(fused)
        return predictions.unsqueeze(-1)  # (batch, pred_length, 1)
    
    def training_step(self, batch, batch_idx):
        if len(batch) == 3:
            x_dynamic, x_static, y = batch
            y_hat = self(x_dynamic, x_static)
        else:
            x_dynamic, y = batch
            y_hat = self(x_dynamic)
        
        loss = nn.functional.mse_loss(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        if len(batch) == 3:
            x_dynamic, x_static, y = batch
            y_hat = self(x_dynamic, x_static)
        else:
            x_dynamic, y = batch
            y_hat = self(x_dynamic)
        
        loss = nn.functional.mse_loss(y_hat, y)
        mae = nn.functional.l1_loss(y_hat, y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_mae', mae, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        if len(batch) == 3:
            x_dynamic, x_static, y = batch
            y_hat = self(x_dynamic, x_static)
        else:
            x_dynamic, y = batch
            y_hat = self(x_dynamic)
        
        loss = nn.functional.mse_loss(y_hat, y)
        mae = nn.functional.l1_loss(y_hat, y)
        self.log('test_loss', loss)
        self.log('test_mae', mae)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }


class EnhancedTimeSeriesDataModule(L.LightningDataModule):
    def __init__(
        self,
        df: pd.DataFrame,
        value_column: str,
        seq_length=96,
        pred_length=1,
        batch_size=32,
        add_time_features=False,
        add_static_features=False
    ):
        super().__init__()
        self.df = df
        self.value_column = value_column
        self.seq_length = seq_length
        self.pred_length = pred_length
        self.batch_size = batch_size
        self.add_time_features = add_time_features
        self.add_static_features = add_static_features

    def setup(self, stage=None):
        self.df = self.df.sort_index()
        data = self.df[self.value_column].values
        timestamps = self.df.index

        test_size = 4 * 24 * 365  # one year of 15-min steps
        train_end = len(data) - test_size - self.pred_length + 1

        train_data = data[:train_end]
        train_ts = timestamps[:train_end]

        test_data = data[train_end - self.seq_length:]
        test_ts = timestamps[train_end - self.seq_length:]

        val_size = int(len(train_data) * 0.1)
        train_data_final = train_data[:-val_size]
        val_data = train_data[-val_size - self.seq_length:]

        train_ts_final = train_ts[:-val_size]
        val_ts = train_ts[-val_size - self.seq_length:]

        self.train_dataset = EnhancedTimeSeriesDataset(
            train_data_final, train_ts_final, self.df, self.value_column,
            self.seq_length, self.pred_length, 
            self.add_time_features, self.add_static_features
        )
        self.val_dataset = EnhancedTimeSeriesDataset(
            val_data, val_ts, self.df, self.value_column,
            self.seq_length, self.pred_length,
            self.add_time_features, self.add_static_features
        )
        self.test_dataset = EnhancedTimeSeriesDataset(
            test_data, test_ts, self.df, self.value_column,
            self.seq_length, self.pred_length,
            self.add_time_features, self.add_static_features
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=0, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=0, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=0, pin_memory=True)

class EnhancedTimeSeriesExperiment:
    def __init__(
        self,
        df: pd.DataFrame,
        value_column,
        seq_length=96,
        pred_length=1,
        batch_size=32,
        wandb_project="timeseries-static-lstm",
        wandb_entity=None,
        add_time_features=False,
        add_static_features=False
    ):
        self.datamodule = EnhancedTimeSeriesDataModule(
            df=df,
            value_column=value_column,
            seq_length=seq_length,
            pred_length=pred_length,
            batch_size=batch_size,
            add_time_features=add_time_features,
            add_static_features=add_static_features
        )
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        self.add_time_features = add_time_features
        self.add_static_features = add_static_features
        
    def _calculate_input_sizes(self):
        dynamic_size = 1  # Base value
        if self.add_time_features:
            dynamic_size += 7  # 7 time features from encode_calendar_features
        
        # Static input size (if enabled)
        static_size = 14 if self.add_static_features else 0
        
        return dynamic_size, static_size
    
    def train_model(
        self,
        hidden_size=64,
        static_embedding_size=32,
        num_layers=2,
        dropout=0.2,
        learning_rate=1e-3,
        max_epochs=100,
        experiment_name=None,
        dev_run_flag=False
    ):
        dynamic_size, static_size = self._calculate_input_sizes()
        
        if experiment_name is None:
            name_parts = [
                f"pred_{self.datamodule.pred_length}",
                f"h{hidden_size}",
                f"l{num_layers}"
            ]
            if self.add_static_features:
                name_parts.append(f"static{static_embedding_size}")
            if self.add_time_features:
                name_parts.append("time")
            experiment_name = "_".join(name_parts)
        
        wandb_logger = WandbLogger(
            project=self.wandb_project,
            entity=self.wandb_entity,
            name=experiment_name,
            log_model=True,
        )
        
        wandb_logger.experiment.config.update({
            "seq_length": self.datamodule.seq_length,
            "pred_length": self.datamodule.pred_length,
            "add_time_features": self.add_time_features,
            "add_static_features": self.add_static_features,
            "dynamic_input_size": dynamic_size,
            "static_input_size": static_size,
        })
        
        model = StaticDynamicLSTM(
            dynamic_input_size=dynamic_size,
            static_input_size=static_size,
            hidden_size=hidden_size,
            static_embedding_size=static_embedding_size,
            num_layers=num_layers,
            dropout=dropout,
            learning_rate=learning_rate,
            pred_length=self.datamodule.pred_length,
        )
        
        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=15,
            mode="min",
            verbose=True
        )
        checkpoint = ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            filename="best-{epoch:02d}-{val_loss:.4f}"
        )
        
        trainer = L.Trainer(
            max_epochs=max_epochs,
            logger=wandb_logger,
            callbacks=[early_stop, checkpoint],
            accelerator="auto",
            devices=1,
            gradient_clip_val=1.0,
            enable_progress_bar=True,
            num_sanity_val_steps=0,
            fast_dev_run=dev_run_flag
        )
        
        trainer.fit(model, datamodule=self.datamodule)
        trainer.test(model, datamodule=self.datamodule, ckpt_path="best", verbose=True)
        
        wandb.finish()
        return model, trainer
    
    def plot_predictions(self, model, n_examples=3):
        model.eval()
        test_loader = self.datamodule.test_dataloader()
        
        example_batches = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                if len(batch) == 3:
                    x_dynamic, x_static, y = batch
                    y_pred = model(x_dynamic, x_static)
                else:
                    x_dynamic, y = batch
                    y_pred = model(x_dynamic)
                
                if batch_idx < n_examples:
                    example_batches.append((x_dynamic.cpu(), y.cpu(), y_pred.cpu()))
                else:
                    break
        
        for i, (x, y, y_pred) in enumerate(example_batches):
            plt.figure(figsize=(12, 5))
            
            # Extract just the value column (first feature)
            input_seq = x[0, :, 0].numpy()
            true_future = y[0].squeeze(-1).numpy()
            pred_future = y_pred[0].squeeze(-1).numpy()
            
            # Plot
            input_range = range(len(input_seq))
            future_range = range(len(input_seq), len(input_seq) + len(true_future))
            
            plt.plot(input_range, input_seq, 'b-', label='Input Sequence', linewidth=2)
            plt.plot(future_range, true_future, 'g-', label='True Future', linewidth=2)
            plt.plot(future_range, pred_future, 'r--', label='Predicted Future', linewidth=2)
            
            # Add vertical line at prediction start
            plt.axvline(x=len(input_seq), color='gray', linestyle=':', alpha=0.7)
            
            plt.title(f"Prediction Example {i+1}" + 
                     (" (with Static Features)" if self.add_static_features else ""),
                     fontsize=14, fontweight='bold')
            plt.xlabel("Time Step", fontsize=12)
            plt.ylabel("Value", fontsize=12)
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
    