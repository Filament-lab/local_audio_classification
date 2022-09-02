import torch
from torch import nn
from torch import optim
from tqdm import tqdm
import numpy as np
import pandas as pd
from typing import List
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from src.model.piczak_cnn import PiczakCnn
from src.utils.config_reader import ConfigReader
from src.utils.custom_error_handler import MLModelException


class ModelTrainer:
    def __init__(self, config: ConfigReader):
        """
        Model selection wrapper
        Add new model to "model_type_map"
        :param config: Config Class
        """
        self.config = config
        self.cnn = PiczakCnn
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Model type selector
        self.model_type_map = {
            "cnn": self.cnn,
        }

    def select_model(self, model_name: str, classes: np.array):
        """
        Select model from pre-defined model map
        :param model_name: Name of model
        :param classes: Classes as numpy array of string
        """
        try:
            self.classes = classes
            self.num_classes = len(classes)
            self.selected_model = self.model_type_map[model_name]
            return self.select_model
        except KeyError:
            raise MLModelException(f"Selected model '{model_name}' does not exist")
        except Exception as err:
            raise MLModelException(f"Error while initializing model: {err}")

    def build(self):
        """
        Initialize model, optimizer, loss function and scheduler
        """
        try:
            self.model = self.selected_model(self.num_classes)
            self.loss_function = nn.CrossEntropyLoss()

            # TODO: Make these configurable
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate, weight_decay=0.0001)
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.1)
        except Exception as err:
            raise MLModelException(f"Error while selecting model: {err}")

    def train(self, train_loader, validation_loader, epoch: int, visualize: bool = True):
        """
        Train model
        :param train_loader: TorchDatasetLoader for training
        :param validation_loader: TorchDatasetLoader for validation
        :param epoch: Number of epoch
        :param visualize: Set to true to visualize train/val accuracy and loss
        """
        train_loss_list, validation_loss_list = [], []
        train_accuracy_list, validation_accuracy_list = [], []
        with tqdm(range(epoch), unit='epoch') as ep:
            ep.set_description('Training')
            # keep track of the running loss
            for epoch in ep:
                self.scheduler.step()
                # Training
                train_loss, train_accuracy = self._calculate_loss_and_accuracy(train_loader, is_train=True)
                train_loss_list.append(train_loss)
                train_accuracy_list.append(train_accuracy)

                # Validation
                validation_loss, validation_accuracy = self._calculate_loss_and_accuracy(validation_loader, is_train=False)
                validation_loss_list.append(validation_loss)
                validation_accuracy_list.append(validation_accuracy)

                print(f'Train Epoch: {epoch+1}\tAccuracy: {train_accuracy:.6f}\tLoss: {train_loss:.6f}')

            # Visualize training result
            if visualize:
                self._draw_curve(train_accuracy_list, train_loss_list, validation_accuracy_list, validation_loss_list)

    def test(self, test_loader, show_confusion_matrix: bool = False):
        self.model.eval()
        correct = 0
        all_predictions = []
        all_labels = []
        for feature, labels in test_loader:
            feature = feature.to(self.device)
            labels = labels.to(self.device)
            output = self.model(feature)
            _, predictions = torch.max(output, 1)
            all_predictions.extend(predictions)
            all_labels.extend(labels)
            correct += (predictions == labels).sum().item()
        print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
            correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

        # Plot confusion matrix
        if show_confusion_matrix:
            ModelTrainer._confusion_matrix(np.array(all_labels), np.array(all_predictions), self.classes)

    def _calculate_loss_and_accuracy(self, dataset_loader, is_train: bool):
        """
        Calculate loss and accuracy for the given dataset
        :param dataset_loader: TorchDataset Loader
        :param is_train: Set to True if it's training dataset
        """
        correct, total = 0, 0
        running_loss = 0
        for batch_idx, (feature, labels) in enumerate(dataset_loader):
            if is_train:
                self.optimizer.zero_grad()
            feature = feature.to(self.device, dtype=torch.float32)
            labels = labels.to(self.device)
            if is_train:
                feature = feature.requires_grad_()  # set requires_grad to True for training

            # Make prediction
            output = self.model(feature)
            _, predictions = torch.max(output, 1)

            # Calculate loss
            loss = self.loss_function(output, labels)
            if is_train:
                loss.backward()
                self.optimizer.step()

            # Calculate loss and accuracy
            running_loss += loss  # add the loss for this batch

            total += labels.size(0)
            correct += (predictions == labels).sum().item()
            total_loss = running_loss/len(dataset_loader)
            total_accuracy = correct/total
            return float(total_loss), float(total_accuracy)

    def _draw_curve(self, train_accuracy_list, train_loss_list, validation_accuracy_list, validation_loss_list):
        """
        Plot train/validation accuracy and loss
        :param train_accuracy_list: List of train accuracy
        :param train_loss_list: List of train loss
        :param validation_accuracy_list: List of validation accuracy
        :param validation_loss_list: List of validation loss
        """
        epochs = range(1, self.config.num_epochs+1)
        plt.plot(epochs, train_accuracy_list, 'g', label='Training accuracy')
        plt.plot(epochs, validation_accuracy_list, 'b', label='Validation accuracy')
        plt.title('Training and Validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()

        plt.plot(epochs, train_loss_list, 'g', label='Training loss')
        plt.plot(epochs, validation_loss_list, 'b', label='Validation loss')
        plt.title('Training and Validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    @staticmethod
    def _confusion_matrix(label: np.ndarray, predictions: np.ndarray, classes: List[str], output_file_path: str = None):
        """
        Show Confusion matrix
        :param label: Ground truth label array
        :param predictions: Model prediction array
        :param classes: List of target classes
        :param output_file_path: Optional output image file path
        """
        # Build confusion matrix
        cf_matrix = confusion_matrix(label, predictions)
        df_cm = pd.DataFrame(cf_matrix.astype('float') / cf_matrix.sum(axis=1)[:, np.newaxis],
                             index=[cl for cl in classes],
                             columns=[cl for cl in classes])
        print(df_cm)
        plt.figure()
        sn.heatmap(df_cm,
                   annot=True,
                   fmt='.2f',
                   cmap='Blues')
        plt.show()

        # Save as file
        if output_file_path:
            plt.savefig(output_file_path)
