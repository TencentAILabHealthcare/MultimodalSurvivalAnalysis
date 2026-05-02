import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)

_pre_train_path = '../pretrained/resnet18-5c106cde.pth'

_pre_train_path = '../models/resnet18-5c106cde.pth'


class ResNet18Feat(nn.Module):
    def __init__(self):
        super(ResNet18Feat, self).__init__()
        base_model = torchvision.models.resnet18(pretrained=False)        
        with open(_pre_train_path, 'rb') as infile:
            state = torch.load(infile, map_location=lambda storage, loc: storage)
            base_model.load_state_dict(state)
        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = base_model.maxpool

        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

class AttentionNetMoreLayer(nn.Module):
    def __init__(self, attentionL=2560, attentionD=128, attentionK=1, dropout_p=0.5, instance_attention_layers=1, instance_attention_dim=128, feature_attention_layers=1, feature_attention_dim=256):
        """
        Args:
            attentionL (int): Number of instance features.
            attentionD (int): Dimensionality of attention.
            attentionK (int): Number of features after attention.
            dropout_p (float): Dropout probability.
            instance_attention_layers (int): Number of layers in instance-level attention network.
            instance_attention_dim (int): Dimensionality of instance-level attention layers.
            feature_attention_layers (int): Number of layers in feature-level attention network.
            feature_attention_dim (int): Dimensionality of feature-level attention layers.
        """
        super(AttentionNetMoreLayer, self).__init__()

        # Initialize hyperparameters
        self.L = attentionL  # Number of instance features
        self.D = attentionD  # Dimensionality of attention
        self.K = attentionK  # Number of features after attention

        # Instance-level attention network
        attention_list = [
            nn.Linear(self.L, instance_attention_dim),
            nn.LeakyReLU(),
        ]
        for _ in range(instance_attention_layers):
            attention_list.extend([
                nn.Dropout(dropout_p),
                nn.Linear(instance_attention_dim, instance_attention_dim),
                nn.BatchNorm1d(instance_attention_dim),
                nn.LeakyReLU(),
            ])
        attention_list.extend([
            nn.Linear(instance_attention_dim, self.K)
        ])
        self.attention = nn.Sequential(*attention_list)

        # Feature-level attention network
        feature_attention_list = [
            nn.Linear(self.L, feature_attention_dim),
            nn.LeakyReLU(),
        ]
        for _ in range(feature_attention_layers):
            feature_attention_list.extend([
                nn.Linear(feature_attention_dim, feature_attention_dim),
                nn.BatchNorm1d(feature_attention_dim),
                Swish()
            ])
        feature_attention_list.extend([
            nn.Linear(feature_attention_dim, self.L)
        ])
        self.feature_attention = nn.Sequential(*feature_attention_list)
        
        # Classifier network
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(self.L * self.K, 1)
        )

    # Get parameters of the instance attention network
    def get_instance_attention_parameters(self):
        params = []
        for param in self.attention.parameters():
            params.append(param)
        return params

    # Get parameters of the feature attention network
    def get_feature_attention_parameters(self):
        params = []
        for param in self.feature_attention.parameters():
            params.append(param)
        return params
    
    # Get parameters of the classifier
    def get_classifier_parameters(self):
        params = []
        for param in self.classifier.parameters():
            params.append(param)
        return params

    # Forward pass through the network
    def forward(self, batch_data: torch.Tensor):
        
        batch_data = batch_data.squeeze(0)
        #print('input',batch_data.shape)
        if len(batch_data.shape) == 2:
            bag = batch_data
        if batch_data.shape[0]==1:#len(batch_data.shape) == 1:
            #bag = batch_data.unsqueeze(0)
            with torch.no_grad():
                bag = torch.cat((bag, bag),0)
        elif len(batch_data.shape) == 1:
            bag = batch_data.unsqueeze(0)
            with torch.no_grad():
                bag = torch.cat((bag, bag),0)
        #print('bag',bag.shape)
        A = self.attention(bag)  # NxK attentions
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        # feature attention
        feature_attention = self.feature_attention(bag)
        feature_attention = torch.sigmoid(feature_attention)
        bag = bag * feature_attention
        M = torch.mm(A, bag)  # KxL
        #print(M.shape)
        M = torch.flatten(M)
        #print(M.shape)
        Y_prob = self.classifier(M)
        #print(Y_prob.shape)
        Y_prob =  torch.sigmoid(Y_prob)
        return Y_prob

    # Get attention weights for visualization
    def get_attention(self, batch_data: torch.Tensor):
        batch_data = batch_data.squeeze(0)
        if len(batch_data.shape) == 2:
            bag = batch_data
        elif len(batch_data.shape) == 1:
            bag = batch_data.unsqueeze(0)
            with torch.no_grad():
                bag = torch.cat((bag, bag),0)

        A = self.attention(bag)
        A = torch.transpose(A, 1, 0)
        A = F.softmax(A, dim=1)

        # feature attention
        feature_attention = self.feature_attention(bag)
        feature_attention = torch.sigmoid(feature_attention)
        bag = bag * feature_attention
        M = torch.mm(A, bag)  # KxL
        M = torch.flatten(M)
        Y_prob = self.classifier(M)
        Y_prob =  torch.sigmoid(Y_prob)  

        return torch.max(A, 0)[0]