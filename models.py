from typing import Any, Union

import torch
import torchvision

class PDModel(torch.nn.Module):
    def __init__(self, base_model, mean_bounded=False, pd_type='Gaussian', eps=1e-2):
        super(PDModel, self).__init__()

        self.base_model = base_model
        self.mean_bounded = mean_bounded
        self.pd_type = pd_type
        self.eps = eps
        print("Bounded", self.mean_bounded, "PD-type", self.pd_type)

    def forward(self, xs):
        output = self.base_model(xs)
        param1 = output[..., 0][..., None] 
        param2 = output[..., 1][..., None]

        if self.pd_type == 'Gaussian':
          mean = torch.clamp(param1, min=0, max=10) if self.mean_bounded else param1 
          std = torch.clamp(param2, min=self.eps)

          dist = torch.distributions.Normal(mean, std)
        return dist

def wrap_PDModel(base_model: torch.nn.Module, **kwargs: Any):
    mean_bounded = kwargs['mean_bounded'] if 'mean_bounded' in kwargs else False 
    pd_type = kwargs['pd_type'] if 'pd_type' in kwargs else "Gaussian"
    if 'eps' in kwargs:
        model = PDModel(base_model, mean_bounded=mean_bounded, eps=kwargs['eps'], pd_type=pd_type)
    else:
        model = PDModel(base_model, mean_bounded=mean_bounded, pd_type=pd_type)
    return model


def load_model(model_name: str, pretrain: Union[None, str], n_labels: int, model_fn: Union[None, str] = None, device: Union[None, torch.device] = None, **kwargs: Any) -> torch.nn.Module:
    train_last_layer_only = kwargs['train_last_layer_only'] if 'train_last_layer_only' in kwargs else False
    if model_name == 'resnet18-b':
        from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn

        const_bnn_prior_parameters = {
                "prior_mu": 0.0,
                "prior_sigma": 1.0,
                "posterior_mu_init": 0.0,
                "posterior_rho_init": -3.0,
                "type": "Reparameterization",  # Flipout or Reparameterization
                "moped_enable": True,  # True to initialize mu/sigma from the pretrained dnn weights
                "moped_delta": 0.5,
                }
        model = torchvision.models.resnet18(weights=torchvision.models.resnet.ResNet18_Weights.IMAGENET1K_V1 if pretrain == "v1" else None)
        dnn_to_bnn(model, const_bnn_prior_parameters)

    elif model_name == 'resnet50':
        assert pretrain in [None, 'V1', 'V2']

        if pretrain == None:       
            weights = None
        elif pretrain == 'V1':     
            weights = torchvision.models.resnet.ResNet50_Weights.IMAGENET1K_V1
        elif pretrain == 'V2':
            weights = torchvision.models.resnet.ResNet50_Weights.IMAGENET1K_V2

        model = torchvision.models.resnet50(weights=weights)
    elif model_name == 'resnet50-b':
        from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn

        const_bnn_prior_parameters = {
                "prior_mu": 0.0,
                "prior_sigma": 1.0,
                "posterior_mu_init": 0.0,
                "posterior_rho_init": -3.0,
                "type": "Reparameterization",  # Flipout or Reparameterization
                "moped_enable": True,  # True to initialize mu/sigma from the pretrained dnn weights
                "moped_delta": 0.5,
                }
        model = torchvision.models.resnet50(weights=torchvision.models.resnet.ResNet18_Weights.IMAGENET1K_V2 if pretrain == "v2" else None)
        dnn_to_bnn(model, const_bnn_prior_parameters)
    
    elif model_name == 'simple-pd':
        from CNNModel import SimpleCNN
        model = SimpleCNN(n_classes=2)
        
        model = wrap_PDModel(model, **kwargs)
    elif model_name == 'resnet18-pd':
        model = torchvision.models.resnet18(weights=torchvision.models.resnet.ResNet18_Weights.IMAGENET1K_V1 if pretrain == 'V1' else None)
        if pretrain is not None and train_last_layer_only:
            for p in model.parameters():
                p.requires_grad = False
        model.fc = torch.nn.Linear(in_features=model.fc.in_features, out_features=2)
        model = wrap_PDModel(model, **kwargs)
    elif model_name == 'resnet50-pd':
        if pretrain == None:       
            weights = None
        elif pretrain == 'V1':     
            weights = torchvision.models.resnet.ResNet50_Weights.IMAGENET1K_V1
        elif pretrain == 'V2':
            weights = torchvision.models.resnet.ResNet50_Weights.IMAGENET1K_V2
        model = torchvision.models.resnet50(weights=weights)

        if pretrain is not None and train_last_layer_only:
            for p in model.parameters():
                p.requires_grad = False
        model.fc = torch.nn.Linear(in_features=model.fc.in_features, out_features=2)
        model = wrap_PDModel(model, **kwargs)
    elif model_name == 'efficientnetv2-s-pd':
        if pretrain == None:       
            weights = None
        elif pretrain == 'V1':     
            weights = torchvision.models.efficientnet.EfficientNet_V2_S_Weights.IMAGENET1K_V1
        model = torchvision.models.efficientnet_v2_s(weights=weights)
        if pretrain is not None and train_last_layer_only:
            for p in model.parameters():
                p.requires_grad = False
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)
        model = wrap_PDModel(model, **kwargs)
    else:
        raise ValueError("Not supported model: {}".format(model))

    return model 
