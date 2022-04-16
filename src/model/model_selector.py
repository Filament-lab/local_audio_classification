import torch
from torch import nn
from torch import optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from src.model.cnn import Net

from src.utils.config_reader import ConfigReader
from src.utils.custom_error_handler import MLModelException


class ModelSelector:
    def __init__(self, config: ConfigReader):
        """
        Feature extraction wrapper
        Add new feature extraction method to "feature_type_map"
        :param config: Config Class
        """
        self.config = config
        self.cnn = Net
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Feature extraction selector
        self.model_type_map = {
            "cnn": self.cnn,
        }

    def select_model(self, model_name: str, num_classes: int):
        """
        Select model from pre-defined model map
        :param model_name: Name of model
        """
        try:
            self.model = self.model_type_map[model_name](num_classes)
            return self.model
        except KeyError:
            raise MLModelException(f"Selected model '{model_name}' does not exist")
        except Exception as err:
            raise MLModelException(f"Error while initializing model: {err}")

    def build(self):
        try:
            self.loss_function = nn.CrossEntropyLoss()
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.01, weight_decay=0.0001)
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.1)
        except Exception as err:
            raise MLModelException(f"Error while selecting model: {err}")

    def train(self, train_loader, epoch: int, visualize: bool = True):
        train_loss, validation_loss = [], []
        train_accuracy, validation_accuracy = [], []
        with tqdm(range(epoch), unit='epoch') as ep:
            ep.set_description('Training')
            # keep track of the running loss
            for epoch in ep:
                self.scheduler.step()
                correct, total = 0, 0
                running_loss = 0.
                for batch_idx, (feature, labels) in enumerate(train_loader):
                    self.optimizer.zero_grad()
                    feature = feature.to(self.device, dtype=torch.float32)
                    labels = labels.to(self.device)
                    feature = feature.requires_grad_()  # set requires_grad to True for training
                    output = self.model(feature)
                    loss = self.loss_function(output, labels)
                    loss.backward()
                    self.optimizer.step()
                    ep.set_postfix(loss=loss.item())
                    running_loss += loss  # add the loss for this batch
                    # get prediction class
                    _, predictions = torch.max(output, 1)
                    total += labels.size(0)
                    correct += (predictions == labels).sum().item()

                epoch_loss = running_loss/len(train_loader)
                epoch_accuracy = correct/total
                train_loss.append(epoch_loss)
                train_accuracy.append(epoch_accuracy)
                print(f'Train Epoch: {epoch}\tAccuracy: {epoch_accuracy:.6f}\tLoss: {epoch_loss:.6f}')

            # Visualize training result
            # if visualize:
            #     plt.plot(train_accuracy, '-o')
            #     # plt.plot(validation_accuracy, '-o')
            #     plt.xlabel('epoch')
            #     plt.ylabel('accuracy')
            #     plt.legend(['Train', 'Validation'])
            #     plt.title('Train vs Validation Accuracy')
            #
            #     plt.show()
            #
            #     plt.plot(train_loss,'-o')
            #     # plt.plot(validation_loss, '-o')
            #     plt.xlabel('epoch')
            #     plt.ylabel('losses')
            #     plt.legend(['Train', 'Valididation'])
            #     plt.title('Train vs Validation Losses')

            # plt.show()
            # TODO: Add validation
            #     model.eval()
            #     running_loss = 0.
            #     correct, total = 0, 0
            #
            #     for data, target in validation_loader:
            #         # getting the validation set
            #         data, target = data.to(device), target.to(device)
            #         optimizer.zero_grad()
            #         output = model(data)
            #         loss = criterion(output, target)
            #         tepochs.set_postfix(loss=loss.item())
            #         running_loss += loss.item()
            #         # get accuracy
            #         _, predicted = torch.max(output, 1)
            #         total += target.size(0)
            #         correct += (predicted == target).sum().item()
            #
            #     validation_loss.append(running_loss/len(validation_loader))
            #     validation_acc.append(correct/total)


    def test(self, test_loader):
        self.model.eval()
        correct = 0
        for feature, labels in test_loader:
            feature = feature.to(self.device)
            labels = labels.to(self.device)
            output = self.model(feature)
            _, predictions = torch.max(output, 1)
            correct += (predictions == labels).sum().item()
        print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
            correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
