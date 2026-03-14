import torch.nn as nn

def get_default_criterions(criterion):
    match(criterion.name):
        case "CrossEntropyLoss":
            return nn.CrossEntropyLoss(label_smoothing=criterion.label_smoothing)
