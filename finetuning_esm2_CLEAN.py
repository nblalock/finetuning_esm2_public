#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing Packages
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.distributed
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from collections import OrderedDict
from torchtext import vocab # This package can give problems sometimes, it may be necesary to downgrade to a specific version
from pytorch_lightning.loggers import CSVLogger
from random import choice
import seaborn as sns
import random
import matplotlib.pyplot as plt
from sklearn import metrics
import torchmetrics
import enum
import argparse
from argparse import ArgumentParser
import os
import pickle
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr
import csv
from coral_pytorch.losses import corn_loss
from coral_pytorch.dataset import corn_label_from_logits
from transformers import AutoModelForMaskedLM, AutoTokenizer


# In[2]:


# import helper scripts
from ESM2_w_regression_MLP_heads_CLEAN import (SeqFcnDataset, ProtDataModule, finetuning_ESM2_with_mse_loss)
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# In[]
# Data parameters

data_filepath = 'gb1.tsv' # ! Change this
df = pd.read_csv(data_filepath, sep='\t')
WT = "MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"
splits_path = None # include if splits stored in a file, else None
splits_type = "num_mutations" # either "file", "num_mutations", or "cluster"

# In[3]

"""
Use this cell to import and pre-process your dataset. Following functions assume there is a "sequence" column for the mutated sequences, and a "score" column.

"""

def mutate(np_mutations: list):
    # 'np_mutations' = list of mutations)
    
    list_updated = []
    count = 0
    
    # Iterates over each element of the input array 'np_mutations'
    for i in range(len(np_mutations)):
        
        # splits the element by ',' (comma) to get the individual mutations.
        try: 
            muts = np_mutations[i].split(',')
        except:
            muts = np_mutations[i]
            
        # Go through each mutation (there are one or two)
        
       # Creates a copy of the original wild type sequence 'WT_list'
        mut_list = list(WT)
        
        # Iterates over each mutation
        for mut in muts:
            
            # nblalock edit: codes extracts the final index and final amino acid from the mutation string
            # The code uses slicing and indexing to extract the information regardless of its length
            final_index = int(mut[1:-1]) - 1
            final_AA = mut[-1]

            # Replaces the amino acid of the wild type sequence with the mutated amino acid
            mut_list[final_index] = final_AA
        
        # Append mutated sequence and score
        list_updated.append(mut_list)
    
    # Returns the list of updated sequences with mutations
    return list_updated


# Fix indexing in variant/mutation entries. This is only necessary if there are issues with 0 v 1 based indexing
def convert_indexing(variants, offset: int):
    """ convert between 0-indexed and 1-indexed """
    #'variants' = an array of strings representing variants/mutations)
    # offset = integer
    
    converted = [",".join(["{}{}{}".format(mut[0], int(mut[1:-1]) + offset, mut[-1]) for mut in v.split(",")])
                 for v in variants]
    # Iterates over each element of the input array 'variants' and for each element
    # Splits the element by ',' (comma) to get the individual mutations
    # Uses list comprehension with "join" method to join the mutated elements with a comma,
    # List comprehension iterates over the individual mutations and for each mutation
    # The first character of the mutation is taken by mut[0]
    # Index value is taken by mut[1:-1] and it converts it to an integer, then it adds the offset value to it
    # the last character of the mutation is taken by mut[-1]
    # Formats it into a string "{}{}{}"
    # First {} will be replaced by the first character of the mutation
    # Second {} will be replaced by the modified index value
    # Third {} will be replaced by the last character of the mutation.
    
    return converted
# The final list comprehension will have a list of modified mutations with the updated indexing,
# then it joins each element of the list using ','(comma) and returns the final list of converted variants/mutations.


# Choose dataset

# Sets the number of threads that PyTorch will use for parallel computation.
torch.set_num_threads(4) 

# The following loads + preprocesses experimentally collected data (fitness scores for gb1 mutants in this example)
data_filepath = 'gb1.tsv' # ! Change this
df = pd.read_csv(data_filepath, sep='\t')
WT = "MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"

# print(df) # shows thermostability data

################################################ May not be necessary ################################################
df.variant = convert_indexing(df.variant,1)    
# print(df) # increases a.a. position by 1
################################################ May not be necessary ################################################

AA_seq_lists = mutate(list(df['variant'].copy()))

AA_seq_lists2 = [str("".join(AA_seq_lists[j])) for j in range(len(AA_seq_lists))]

# Add column of full amino acid sequences.
df['Sequence'] = AA_seq_lists2

print(f"Using dataset from {data_filepath}")

df = df.rename(columns={ "Sequence" : "sequence"})
df['class'] = df['score'].apply(lambda x: 0 if x < 0 else 1) # 0 = "dead", 1 = "functional"

# Plot histogram
plt.figure(figsize=(10, 6))
plt.hist(df['score'], bins=50, edgecolor='black', alpha=0.7)
plt.xlabel('Score')
plt.ylabel('Frequency')
plt.title('Histogram of Scores')
plt.grid(axis='y', alpha=0.75)

# Show plot
plt.show()


# In[4]:


# Choose label strategy

# Define reg_target_labels to use
reg_target_labels = [
    'score'
    #'Titer (mg/L)',
    # 'Background A280',
    # 'aSEC % Area Main Peak',
    # 'aSEC % Area HMW',
    # 'aSEC % Area LMW',
    # 'DSF Fc Unfold (°C)',
    # 'DSF ADA Unfold (°C)', 
    # 'Mean Fold Change'
    ] # ! update

log_target_labels = [
    "class"
] # ! update

ordinal_reg_target_labels = [
    # 'aSEC Retention Time (Main Peak)',
    ] # ! update

# Dynamically find reg_target_labels indices
reg_target_labels_indices = [df.columns.get_loc(reg_target_labels) for reg_target_labels in reg_target_labels if reg_target_labels in df.columns]

log_target_labels_indices = [df.columns.get_loc(log_target_labels) for log_target_labels in log_target_labels if log_target_labels in df.columns]

ordinal_reg_target_labels_indices = [df.columns.get_loc(reg_target_labels) for reg_target_labels in ordinal_reg_target_labels if reg_target_labels in df.columns]

# Determine the number of logistic classes per label
num_log_classes = [df[label].nunique() for label in log_target_labels if label in df.columns]

# Determine the number of unique variables for ordinal regression labels
ordinal_reg_target_nunique = [df[label].nunique() for label in ordinal_reg_target_labels if label in df.columns]


print(f"Regressing {reg_target_labels} at indices {reg_target_labels_indices}")
print(f"Classifying {log_target_labels} at indices {log_target_labels_indices}")
print(f"Ordinally regressing {ordinal_reg_target_labels} at indices {ordinal_reg_target_labels_indices}")
print(f"Number of unique variables in ordinal regression labels: {ordinal_reg_target_nunique}")


# In[6]:
######################################## Hyperparameters that can be altered ########################################
# ESM2 selection
huggingface_identifier ='esm2_t6_8M_UR50D' # esm2_t6_8M_UR50D # esm2_t12_35M_UR50D # esm2_t30_150M_UR50D # esm2_t33_650M_UR50D
ESM2 = AutoModelForMaskedLM.from_pretrained(f"facebook/{huggingface_identifier}")
tokenizer = AutoTokenizer.from_pretrained(f"facebook/{huggingface_identifier}")
model_identifier = huggingface_identifier
token_format = 'ESM2'

# Model training hyperparameters
num_unfrozen_layers = 0
num_layers_unfreeze_each_epoch = 15
max_num_layers_unfreeze_each_epoch = 36
epoch_threshold_to_unlock_ESM2 = 100
hidden_layer_size_1 = 300 # ! update
hidden_layer_size_2 = 5 # ! update

# Learning hyperparameters
epochs = 50 # ! update
patience = 10 # ! update
warm_restart = 1 # with warm restart
use_scheduler = 1 # with scheduler
WD = 0.005
grad_clip_threshold = 3.0
lr_mult = 1
lr_mult_factor = 1
seed = 3
learning_rate = 1e-6 # ! update
reinit_optimizer = 0
using_EMA = 1
decay = 0.8

# GPU hyperparameters
embedding_type = 'all_tokens' # 'all_tokens' # ! Change this
if embedding_type == 'cls_token_only':
    batch_size = 32 # typically powers of 2: 32, 64, 128, 256, ...
elif embedding_type == 'mean_pooling' or embedding_type == 'max_pooling':
    batch_size = 32 # typically powers of 2: 32, 64, 128, 256, ...
elif embedding_type == 'all_tokens':
    batch_size = 16
else:
    print('Isue selecting embedding type')

# Data hyperparameters
slen = len(WT) # length of protein
num_reg_tasks = 1 # len(reg_target_labels)
reg_weights = [1] # ! update
num_log_tasks = len(log_target_labels)
reg_type = ['mse'] # ["mse", "log", "ord"] include any of the 3 based on task
num_ord_reg_tasks = 0 # len(ordinal_reg_target_labels)
ord_reg_weights = [] # ! update
ord_reg_type = "corn_loss"

filepath = f'finetuning_ESM2_with_{data_filepath}'

# Determine if we're running on a GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Determine if we're running on a GPU
if device == "cuda":
    # Make models reproducible on GPU
    os.environ['PYTHONHASHSEED'] = str(seed) # Set the PYTHONHASHSEED environment variable to the chosen seed to make hash-based operations predictable
    np.random.seed(seed) # Set NumPy's random seed to ensure reproducibility of operations using NumPy's random number generator
    random.seed(seed) # Set Python's built-in random module's seed to ensure reproducibility of random operations using Python's random functions
    np.random.seed(seed)
    torch.manual_seed(seed) # Set the seed for generating random numbers in PyTorch to ensure reproducibility on the CPU
    torch.cuda.manual_seed(seed) # Set the seed for generating random numbers in PyTorch to ensure reproducibility on the GPU
    torch.cuda.manual_seed_all(seed) # Ensure reproducibility for all GPUs by setting the seed for generating random numbers for all CUDA devices
    torch.backends.cudnn.deterministic = True # Force cuDNN to use only deterministic convolutional algorithms (can slow down computations but guarantees reproducibility)
    torch.backends.cudnn.benchmark = False # Prevent cuDnn from using any algorithms that are nondeterministic
    torch.set_float32_matmul_precision('medium')
    print('Training model on GPU')
else:
    # fix random seeds for reproducibility on CPU
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    print('Training model on CPU')

######################################## Hyperparameters that can be altered ########################################


# In[7]:

# Data Module

dm = ProtDataModule(df, reg_target_labels_indices, log_target_labels_indices, ordinal_reg_target_labels_indices, batch_size, splits_path, splits_type, token_format, seed)


# In[8]:

model = finetuning_ESM2_with_mse_loss(ESM2, huggingface_identifier, tokenizer, num_unfrozen_layers, num_layers_unfreeze_each_epoch, max_num_layers_unfreeze_each_epoch, hidden_layer_size_1, hidden_layer_size_2,
                 epochs, batch_size, seed, embedding_type, patience,
                 learning_rate, lr_mult, lr_mult_factor,
                 WD, reinit_optimizer, grad_clip_threshold, use_scheduler, warm_restart,
                 slen, num_reg_tasks, num_log_classes, num_log_tasks, reg_weights, reg_type, num_ord_reg_tasks, ord_reg_weights, None, ordinal_reg_target_nunique,
                 using_EMA, decay,
                 epoch_threshold_to_unlock_ESM2,
                 WT,
                 data_filepath)

checkpoint_callback = ModelCheckpoint(
        dirpath=f"./logs/{filepath}/",
        filename=f"{filepath}",
        monitor="val_reg_loss",
        mode="min",
        save_top_k=1)
early_stopping = EarlyStopping(monitor="val_reg_loss", patience=patience, mode="min")
logger = CSVLogger('logs', name=f"{filepath}") # logger is a class instance that stores performance data to a csv after each epoch

# Dynamically set up Trainer based on available device
trainer = pl.Trainer(
    logger=logger,
    max_epochs=epochs,
    callbacks=[checkpoint_callback, early_stopping],
    enable_progress_bar=True,
    accelerator=device,  # Automatically chooses between "cpu" and "gpu"
    devices=1 if device == "cuda" else "auto",  # Use 1 GPU if available, else default to CPU
    deterministic=True  # Ensure reproducibility
)
#try:
trainer.fit(model, dm)
#except Exception as e:
 #   print(f"Training stopped due to an error: {e}")


# In[9]:


# Save the model
non_ema_path = f'./logs/{filepath}/version_{logger.version}/ESM2_CreiLOV.pt'
ema_path = f'./logs/{filepath}/version_{logger.version}/ESM2__CreiLOV_w_EMA.pt'
model.save_model(non_ema_path, ema_path)


# In[10]:


################################################################################################################################################

# make learning curves
version = logger.version  # Replace `logger.version` with the specific version number if needed
train_losses = []
val_losses = []

# Load the metrics for the specified version
try:
    # Read metrics.csv for the specified version
    pt_metrics = pd.read_csv(f'./logs/{filepath}/version_{version}/metrics.csv')
    
    # Extract training and validation losses
    train = pt_metrics[~pt_metrics.train_reg_loss.isna()]
    val = pt_metrics[~pt_metrics.val_reg_loss.isna()]
    train_losses = train.train_reg_loss.values
    val_losses = val.val_reg_loss.values
except FileNotFoundError:
    print(f"Metrics file for version {version} not found.")
    train_losses = []
    val_losses = []

# Check if losses are available
if len(train_losses) > 0 and len(val_losses) > 0:
    # Ensure losses have the same length by padding if necessary
    max_length = max(len(train_losses), len(val_losses))
    train_losses = np.pad(train_losses, (0, max_length - len(train_losses)), 'constant', constant_values=np.nan)
    val_losses = np.pad(val_losses, (0, max_length - len(val_losses)), 'constant', constant_values=np.nan)

    # Compute epochs
    epochs = np.arange(1, max_length + 1)

    # Plot the loss curves
    plt.plot(epochs, train_losses, label='training loss')
    plt.plot(epochs, val_losses, label='validation loss')
    plt.title('Loss vs. Epoch')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()

    # Save the loss curves
    file_path_svg = os.path.join(f'./logs/{filepath}/version_{version}', 'Loss_Curves.svg')
    plt.savefig(file_path_svg)
    file_path_png = os.path.join(f'./logs/{filepath}/version_{version}', 'Loss_Curves.png')
    plt.savefig(file_path_png)
    print(f"Loss curves saved to {file_path_svg} and {file_path_png}")
else:
    print("No loss data found for this model version.")


# In[11]:


################################################################################################################################################

checkpoint_path = ema_path

# Initialize dictionaries to store regression predictions and ordinal labels for train, validation, and test sets
reg_values_train = {idx: [] for idx in dm.train_idx}
reg_values_val = {idx: [] for idx in dm.val_idx}
reg_values_test = {idx: [] for idx in dm.test_idx}
ord_probs_train = {idx: [] for idx in dm.train_idx}
ord_probs_val = {idx: [] for idx in dm.val_idx}
ord_probs_test = {idx: [] for idx in dm.test_idx}

# Load the saved model checkpoint
model = finetuning_ESM2_with_mse_loss(ESM2, huggingface_identifier, tokenizer, num_unfrozen_layers, num_layers_unfreeze_each_epoch, max_num_layers_unfreeze_each_epoch, hidden_layer_size_1, hidden_layer_size_2,
                 epochs, batch_size, seed, embedding_type, patience,
                 learning_rate, lr_mult, lr_mult_factor,
                 WD, reinit_optimizer, grad_clip_threshold, use_scheduler, warm_restart,
                 slen, num_reg_tasks, reg_weights, reg_type, num_ord_reg_tasks, ord_reg_weights, ord_reg_type, ordinal_reg_target_nunique,
                 using_EMA, decay,
                 epoch_threshold_to_unlock_ESM2,
                 WT,
                 data_filepath)

checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint)
model.eval()

# Split data into train and validation sets
df = pd.read_pickle(f"./{data_filepath}").reset_index(drop=True) # load preprocessed CreiLOV data
train_df, val_df, test_df = df.iloc[dm.train_idx], df.iloc[dm.val_idx], df.iloc[dm.test_idx]

# Set batch size
batch_size = 32  # You can experiment with different batch sizes for optimal speed and memory usage

# Prediction loop for train, validation, and test sets
for data_frame, reg_values_store, ord_probs_store, dataset_name in zip(
    [train_df, val_df, test_df],
    [reg_values_train, reg_values_val, reg_values_test],
    [ord_probs_train, ord_probs_val, ord_probs_test],
    ["train", "validation", "test"],
):
    sequences = data_frame['Sequence'].tolist()  # Extract all sequences
    for start_idx in range(0, len(sequences), batch_size):
        end_idx = min(start_idx + batch_size, len(sequences))
        batch_sequences = sequences[start_idx:end_idx]

        # Predict using the model
        reg_preds, ord_probs = model.predict(batch_sequences)  # Call updated predict function

        # Store predictions
        for i, idx in enumerate(data_frame.index[start_idx:end_idx]):
            reg_values_store[idx] = reg_preds[i].astype(float)  # Store regression predictions
            ord_probs_store[idx] = [
                task_probs[i].astype(float) for task_probs in ord_probs
            ]  # Store ordinal probabilities for each task

    print(f"Processed all sequences in {dataset_name} set.")


# In[12]:


################################################################################################################################################

if num_reg_tasks > 0:

    # Prepare actual and predicted values for all regression labels
    X_reg_train = train_df[reg_target_labels].values  # Actual values for train set
    X_reg_val = val_df[reg_target_labels].values  # Actual values for validation set
    X_reg_test = test_df[reg_target_labels].values  # Actual values for test set

    Y_reg_train = np.array([reg_values_train[idx] for idx in train_df.index])  # Predicted values for train set
    Y_reg_val = np.array([reg_values_val[idx] for idx in val_df.index])  # Predicted values for validation set
    Y_reg_test = np.array([reg_values_test[idx] for idx in test_df.index])  # Predicted values for test set

    # Number of labels and rows/columns for subplots
    num_labels = len(reg_target_labels)
    rows = (num_labels + 2) // 3  # Arrange in 3 columns
    cols = 3

    # Create the figure and subplots
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 5))
    axes = axes.flatten()  # Flatten for easy indexing if rows > 1

    # Iterate over regression labels and plot on subplots
    for i, label in enumerate(reg_target_labels):
        ax = axes[i]

        # Replace spaces in the label with underscores for safe filenames
        label_safe = label.replace(" ", "_")
        
        # Remove NaN values
        valid_mask = ~np.isnan(X_reg_val[:, i]) & ~np.isnan(Y_reg_val[:, i])
        X_reg_val_clean = X_reg_val[:, i][valid_mask]
        Y_reg_val_clean = Y_reg_val[:, i][valid_mask]

        # Plot actual vs. predicted for training, validation, and test sets
        ax.scatter(X_reg_train[:, i], Y_reg_train[:, i], color='blue', s=5, label="Train")
        ax.scatter(X_reg_val_clean, Y_reg_val_clean, color='orange', s=5, label="Validation")
        ax.scatter(X_reg_test[:, i], Y_reg_test[:, i], color='red', s=5, label="Test")
        ax.plot([X_reg_train[:, i].min(), X_reg_train[:, i].max()],
                [X_reg_train[:, i].min(), X_reg_train[:, i].max()], color='black', linestyle='--', linewidth=0.5)
        
        # Add axis labels, legend, and title
        ax.set_xlabel(f"{label} (Actual)", fontsize=10)
        ax.set_ylabel(f"{label} (Predicted)", fontsize=10)
        ax.legend(loc='upper right')
        ax.set_title(f"Predicted vs. Actual {label}")

        # Calculate and annotate metrics on validation set for label i
        if len(X_reg_val_clean) > 0:  # Ensure there are valid entries
            mse = metrics.mean_squared_error(X_reg_val_clean, Y_reg_val_clean)
            r = np.corrcoef(X_reg_val_clean, Y_reg_val_clean)[0][1]
            rho, _ = spearmanr(X_reg_val_clean, Y_reg_val_clean)
            ax.text(0.05, 0.95, f"MSE = {mse:.2f}", fontsize=10, transform=ax.transAxes)
            ax.text(0.05, 0.9, f"R = {r:.2f}", fontsize=10, transform=ax.transAxes)
            ax.text(0.05, 0.85, f"Rho = {rho:.2f}", fontsize=10, transform=ax.transAxes)
        else:
            ax.text(0.05, 0.9, "No valid data", fontsize=10, transform=ax.transAxes)

    # Hide unused subplots if any
    for j in range(num_labels, len(axes)):
        fig.delaxes(axes[j])

    # Adjust layout and save the figure
    fig.tight_layout()
    os.makedirs(f'./logs/{filepath}', exist_ok=True)
    fig.savefig(f'./logs/{filepath}/version_{logger.version}/ema_regression_predictions_all_labels_w_EMA_model.png')
    fig.savefig(f'./logs/{filepath}/version_{logger.version}/ema_regression_predictions_all_labels_w_EMA_model.svg')

    # Show the figure
    plt.show()


# In[13]:


################################################################################################################################################

if num_ord_reg_tasks > 0:

    # Define splits for train, validation, and test
    splits = {
        "train": ord_probs_train,
        "validation": ord_probs_val,
        "test": ord_probs_test,
    }

    # Initialize dictionaries to store individual MAE values
    individual_mae = {split: {label: [] for label in ordinal_reg_target_labels} for split in splits}

    # Iterate through splits and compute MAE for each data point
    for split_name, split_data in splits.items():
        if split_name == 'test':
            task_labels = {label: np.array([test_df[label][key] for key in split_data.keys()])
                           for label in ordinal_reg_target_labels}
        elif split_name == 'validation':
            task_labels = {label: np.array([val_df[label][key] for key in split_data.keys()])
                           for label in ordinal_reg_target_labels}
        elif split_name == 'train':
            task_labels = {label: np.array([train_df[label][key] for key in split_data.keys()])
                           for label in ordinal_reg_target_labels}

        # Compute MAE for each data point
        for i, key in enumerate(split_data.keys()):
            for label_idx, label in enumerate(ordinal_reg_target_labels):
                # Extract probabilities for the current task
                task_probs = split_data[key][label_idx]  # Probabilities for the current task
                true_label = task_labels[label][i]  # True label for the current task

                # Compute predicted probabilities for the true label
                pred_prob = task_probs[true_label]

                # Compute the absolute error and store it
                individual_mae[split_name][label].append(abs(1 - pred_prob))

    # Plotting the results
    fig, axes = plt.subplots(1, len(ordinal_reg_target_labels), figsize=(12, 6), sharey=True)

    # Iterate over each task
    for task_idx, label in enumerate(ordinal_reg_target_labels):
        # Calculate average MAE for validation set
        avg_mae_validation = np.mean(individual_mae["validation"][label])

        # Scatter plot for each split
        for split_name, color in zip(["train", "validation", "test"], ["blue", "orange", "red"]):
            x_bins = []
            for i, key in enumerate(splits[split_name].keys()):
                if split_name == "train":
                    x_bins.append(train_df[label][key])
                elif split_name == "validation":
                    x_bins.append(val_df[label][key])
                elif split_name == "test":
                    x_bins.append(test_df[label][key])

            axes[task_idx].scatter(x_bins,
                                   individual_mae[split_name][label],
                                   color=color, label=f"{split_name.capitalize()} MAE", alpha=0.7)

        # Annotate average MAE for validation set
        axes[task_idx].annotate(f"Avg MAE (Validation): {avg_mae_validation:.4f}",
                                xy=(0.05, 0.95), xycoords="axes fraction",
                                fontsize=10, backgroundcolor="white",
                                verticalalignment="top")

        # Set plot title and labels
        axes[task_idx].set_xlabel("Bin (True Label)")
        if task_idx == 0:
            axes[task_idx].set_ylabel("MAE")
        axes[task_idx].legend(loc='upper right')

    plt.tight_layout()
    os.makedirs(f'./logs/{filepath}', exist_ok=True)
    fig.savefig(f'./logs/{filepath}/version_{logger.version}/ema_ordinal_mae_w_EMA_model.png')
    fig.savefig(f'./logs/{filepath}/version_{logger.version}/ema_ordinal_mae_w_EMA_model.svg')

    # Show the figure
    plt.show()




# In[ ]:




