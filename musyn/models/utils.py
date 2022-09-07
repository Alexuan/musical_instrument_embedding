import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from musyn.models.criterion import AngleLoss, CrossEntropyLoss, NLLLoss


###############################################################################
# Optimizer
###############################################################################

def get_optimizer(cfg, model):
    if cfg.type == 'Adam':
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.lr, betas=cfg.betas, eps=cfg.eps, 
            weight_decay=cfg.weight_decay, amsgrad=True)
    else:
        return NotImplementedError('optimizer [%s] is not implemented', opt.type) 
    
    return optimizer


###############################################################################
# LR_Scheduler
###############################################################################

class ScheduledOptim(object):
    """ A simple wrapper class for learning rate scheduling """

    def __init__(self, optimizer, n_warmup_steps):
        self.optimizer = optimizer
        self.d_model = 64
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.delta = 1

    def increase_delta(self):
        self.delta *= 2

    def step(self):
        "Learning rate scheduling per step"
        self.n_current_steps += self.delta
        new_lr = np.power(self.d_model, -0.5) * np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        return new_lr

    def state_dict(self):
        ret = {
            'd_model': self.d_model,
            'n_warmup_steps': self.n_warmup_steps,
            'n_current_steps': self.n_current_steps,
            'delta': self.delta,
        }
        ret['optimizer'] = self.optimizer.state_dict()
        return ret

    def load_state_dict(self, state_dict):
        self.d_model = state_dict['d_model']
        self.n_warmup_steps = state_dict['n_warmup_steps']
        self.n_current_steps = state_dict['n_current_steps']
        self.delta = state_dict['delta']
        self.optimizer.load_state_dict(state_dict['optimizer'])


def get_scheduler(cfg, optimizer):

    if cfg.type == 'ScheduledOptim':
        scheduler =  ScheduledOptim(
            optimizer, n_warmup_steps=cfg.n_warmup_steps
        )
    elif cfg.type == 'CyclicLR':
        scheduler = optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=cfg.base_lr,
            max_lr=cfg.max_lr,
            step_size_up=cfg.step_size_up,
            step_size_down=cfg.step_size_down,
            mode=cfg.mode,
            cycle_momentum=False
        )
    return scheduler


###############################################################################
# Criterion
###############################################################################


def get_criterion(cfg, index=None):
    if isinstance(cfg.type, list):
        crite_type = cfg.type[index]
    else:
        crite_type = cfg.type

    if crite_type == 'AngleLoss': 
        # angular-softmax
        criterion = AngleLoss()
        print('trained with Angular Softmax')
    elif crite_type == 'NLL':
        criterion = NLLLoss(reduction='mean')
        print('trained with NLL Loss')
    elif crite_type == 'CrossEntropy':
        criterion = CrossEntropyLoss(reduction='mean')
        print('trained with Cross Entropy Loss')
    elif crite_type == 'MSE':
        criterion = nn.MSELoss(reduction='mean')
        print('trained with MSE')
    elif crite_type == 'BCE':
        criterion = nn.BCELoss(reduction='mean')
        print('trained with BCE')
    elif crite_type == 'BCEWithLogits':
        criterion = nn.BCEWithLogitsLoss(reduction='mean')
        print('trained with BCEWithLogits')
    else:
        return NotImplementedError('criterion [%s] is not implemented', crite_type)
    
    return criterion
        