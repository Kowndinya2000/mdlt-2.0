import torch
import torch.nn as nn
import sys
import os 
sys.path.append('/home/exx/Project/ws_Kowndinya/benchmarking/ext/mdlt-2.0/mdlt/models/external')
from miro.domainbed.optimizers import get_optimizer
from miro.domainbed.networks.ur_networks import URFeaturizer


class ModifiedMIRO(nn.Module):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ModifiedMIRO, self).__init__()
        print('num_domains: ', num_domains)
        print('num_classes: ', num_classes)
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.num_domains = num_domains
        self.hparams = hparams
        
        self.pre_featurizer = URFeaturizer(
            input_shape, self.hparams, freeze="all", feat_layers=hparams["feat_layers"]
        )
        self.featurizer = URFeaturizer(
            input_shape, self.hparams, feat_layers=hparams.feat_layers
        )
        self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.ld = hparams.ld

        # build mean/var encoders
        shapes = self.get_shapes(self.pre_featurizer, self.input_shape)
        self.mean_encoders = nn.ModuleList([
            MeanEncoder(shape) for shape in shapes
        ])
        self.var_encoders = nn.ModuleList([
            VarianceEncoder(shape) for shape in shapes
        ])

        # optimizer
        parameters = [
            {"params": self.network.parameters()},
            {"params": self.mean_encoders.parameters(), "lr": hparams.lr * hparams.lr_mult},
            {"params": self.var_encoders.parameters(), "lr": hparams.lr * hparams.lr_mult},
        ]
        self.optimizer = get_optimizer(
            hparams["optimizer"],
            parameters,
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )

    def forward(self, x):
        return self.predict(x)

    def get_shapes(self, model, input_shape):
        # get shape of intermediate features
        with torch.no_grad():
            dummy = torch.rand(1, *input_shape).to(next(model.parameters()).device)
            _, feats = model(dummy, ret_feats=True)
            shapes = [f.shape for f in feats]

        return shapes

    # TODO: We can implement the update method here as per miro paper or can just leave it as it is
    # def update(self, x, y, **kwargs):
    #     pass

    def predict(self, x):
        return self.network(x)

    def get_forward_model(self):
        forward_model = ForwardModel(self.network)
        return forward_model


# # Usage example:
# hparams = {
#     'feat_layers': [1, 2],
#     'ld': 0.1,
#     'lr': 5e-05,
#     'lr_mult': 1,
#     'optimizer': 'Adam',
#     'weight_decay': 0.0
# }

# input_shape = (3, 224, 224)
# num_classes = 10
# num_domains = 3

# model = ModifiedMIRO(input_shape, num_classes, num_domains, hparams)
