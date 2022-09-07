import datetime
import os
import os.path as osp
import random
import shutil
import time

import numpy as np
import torch
from musyn.data import create_dataset
from musyn.engine.base_trainer import BaseTrainer
from musyn.evaluation import compute_eer_det, plot_TSNE
from musyn.models import (create_model, get_criterion, get_optimizer,
                          get_scheduler)
from musyn.utils import mkdir_or_exist


class VanillaTrainer(BaseTrainer):
    """A simple trainer class implementing generic functions."""

    def __init__(self, cfg):
        super().__init__()
        
        torch.manual_seed(cfg.seed)
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.cuda.manual_seed_all(cfg.seed)
        else:
            self.device = torch.device('cpu')
        print('use cuda is %s' % torch.cuda.is_available())

        # Save as attributes some frequently used variables
        self.start_epoch = self.epoch = 0
        self.max_epoch = cfg.runner.max_epochs
        self.output_dir = cfg.log_config.dir
        mkdir_or_exist(cfg.log_config.dir)

        self.cfg = cfg
        self.build_data_loader()
        self.build_model()


    def build_data_loader(self):
        """Create essential data-related attributes.

        What must be done in the re-implementation
        of this method:
        1) initialize data manager
        2) assign as attributes the data loaders
        3) assign as attribute the number of classes
        """
        cfg = self.cfg
        self.dataset_train = create_dataset(cfg.data_train)
        print('The number of training samples = %d' % len(self.dataset_train))
        self.dataset_valid = create_dataset(cfg.data_valid)
        print('The number of valid samples = %d' % len(self.dataset_valid))
        self.dataset_test = create_dataset(cfg.data_test)
        print('The number of test samples = %d' % len(self.dataset_test))


    def build_model(self):
        """Build model.

        The default builds a classification model along with its
        optimizer and scheduler.

        Custom trainers can re-implement this method if necessary.
        """
        cfg = self.cfg
        print('Building model')
        self.model = create_model(cfg.model)
        self.model.setup(cfg.model)

        if cfg.pretrain_model_pth is not None:
            self.load_pretrained_weights()
        self.model.to(self.device)

        self.optim = get_optimizer(cfg.optimizer, self.model)
        self.sched = get_scheduler(cfg.optimizer.lr_scheduler, self.optim)
        self.criterion = get_criterion(cfg.criterion)


    def load_pretrained_weights(self):
        cfg = self.cfg
        if os.path.isfile(cfg.pretrain_model_pth):
            print('loading pre-trained model from %s' % cfg.pretrain_model_pth)
            checkpoint = torch.load(cfg.pretrain_model_pth, 
                map_location=lambda storage, loc: storage) # load for cpu
            self.model.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            print("===> no checkpoint found at '{}'".format(cfg.pretrain_model_pth))


    def train(self):
        super().train()


    def before_train(self):
        log_dir = os.path.join(self.cfg.log_config.dir, 'run')
        self.epoch = 0
        self.n_iter = 0
        self.best_acc = 0
        self.best_eer = 1
        self.init_writer(log_dir)
        self.time_start = time.time()


    def run_epoch(self):
        cfg = self.cfg
        device = self.device
        num_samples = 0
        self.model.train()
        for batch_idx, data in enumerate(self.dataset_train):
            # before step
            data_A = data['feat'].to(device)
            label_A = data['label_A'].to(device)
            label_B = data['label_B'].to(device)
            # run_step
            self.optim.zero_grad()
            _, output_A, output_B = self.model(data_A)
            loss_A = self.criterion(output_A, label_A)
            loss_B = self.criterion(output_B, label_B)
            loss = cfg.loss_weight.label_A * loss_A + cfg.loss_weight.label_B * loss_B
            loss.backward()
            self.optim.step()
            self.sched.step()
            lr = self.sched.step()
            # after step
            self.write_scalar('LearningRate/train', lr, self.n_iter)
            self.write_scalar('Loss/train', loss, self.n_iter)
            self.write_scalar('Loss/train_label_A', loss_A, self.n_iter)
            self.write_scalar('Loss/train_label_B', loss_B, self.n_iter)
            self.n_iter += 1
            num_samples += data_A.shape[0]

            if batch_idx % cfg.log_config.interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tlr:{:.5f}\tLoss: {:.6f}'.format(
                    self.epoch, num_samples, len(self.dataset_train),
                    100. * num_samples / len(self.dataset_train), lr, loss.item()))


    def after_epoch(self):
        meet_test_freq = self.epoch % self.cfg.log_config.interval == 0
        if meet_test_freq is False:
            pass
        else:
            self.valid()
            self.save_model()


    def after_train(self):
        print('Finished training')

        # Save model
        self.save_model()

        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print('Elapsed: {}'.format(elapsed))

        # Close writer
        self.close_writer()


    @torch.no_grad()
    def valid(self):
        cfg = self.cfg
        """A generic testing pipeline."""
        self.model.eval()

        test_loss_B = 0
        feats_A = []
        labels_A = []
        labels_B = []

        for data in self.dataset_valid:
            data_A = data['feat'].to(self.device)
            label_A = data['label_A'].to(self.device)
            label_B = data['label_B'].to(self.device) 
            feat_A, _, output_B = self.model(data_A)
            
            test_loss_B += self.criterion(output_B, label_B).item()
            if cfg.model.head2.type == 'AngularClsHead':
                output_B = output_B[0]
            pred_B = output_B.max(1, keepdim=True)[1] # get the index of the max log-probability
            
            feats_A.append(feat_A.detach().cpu().numpy())
            labels_A.append(label_A.detach().cpu().numpy())
            labels_B.append(label_B.view_as(pred_B))

        feats_A = np.vstack(feats_A)
        labels_A = np.hstack(labels_A)
        labels_B = torch.vstack(labels_B)

        # record test loss & accuracy for instrument
        test_loss_B = test_loss_B / len(self.dataset_valid) * cfg.data_valid.batch_size
        eer = compute_eer_det(feats_A, labels_A, save_dir=self.cfg.log_config.dir)
        self.write_scalar('Loss/test_label_B', test_loss_B, self.epoch)
        self.write_scalar('EER/test_label_A', eer, self.epoch)
        print('\nTest-Lable-B: Average loss: {:.4f}, EER: {:.4f}\n'.format(
            test_loss_B, eer))

        self.is_save_model = True if eer < self.best_eer else False
        self.best_eer = eer if eer < self.best_eer else self.best_eer


    @torch.no_grad()
    def inference(self):
        cfg = self.cfg
        log_dir = os.path.join(cfg.log_config.dir, 'run_inference')
        self.init_writer(log_dir)

        self.model.eval()
        feats_A = []
        labels_A = []
        labels_B = []

        for _, data in enumerate(self.dataset_test):
            data_A = data['feat'].to(self.device)
            label_A = data['label_A'].to(self.device)
            label_B = data['label_B'].to(self.device)
            
            feat_A, _, _ = self.model(data_A)

            feats_A.append(feat_A.detach().cpu().numpy())
            labels_A.append(label_A.detach().cpu().numpy())
            labels_B.append(label_B.detach().cpu().numpy())
        
        feats_A = np.vstack(feats_A)
        labels_A = np.hstack(labels_A)
        labels_B = np.hstack(labels_B)

        ### Evaluation Matrix
        # EER
        _ = compute_eer_det(feats_A, labels_A, save_dir=self.cfg.log_config.dir)


    @torch.no_grad()
    def decode(self):
        self.model.eval()
        mkdir_or_exist(osp.join(self.cfg.decode.dir, 'feat'))

        for i, data in enumerate(self.dataset_test):
            data_A = data['feat'].to(self.device)
            feat_A = self.model.predict(data_A)
            feat_A = feat_A.detach().cpu().numpy()
            feat_name = self.dataset_test.dataset.datalist[i]
            feat_A.tofile(osp.join(self.cfg.decode.dir, 'feat', feat_name + '.npy'),
                          format='<f4')


    def get_current_lr(self, names=None):
        names = self.get_model_names(names)
        name = names[0]
        return self._optims[name].param_groups[0]['lr']


    def save_model(self, mode='eer'):
        cfg = self.cfg
        epoch = self.epoch
        acc = self.best_acc
        eer = self.best_eer * 100
        result = eer if mode == 'eer' else acc
        model_dir = osp.join(cfg.log_config.dir, 
            str(epoch) + "_" + str(int(result)) + ".h5")
        best_model_dir = osp.join(cfg.log_config.dir, 
            'model_best.pth.tar')
        if self.is_save_model is True:
            torch.save({
                'epoch': self.epoch,
                'state_dict': self.model.state_dict(),
                'best_result': result,
                'optimizer' : self.optim.state_dict(),
            }, model_dir)
            print("===> save to checkpoint at {}\n".format(best_model_dir))
            shutil.copyfile(model_dir, best_model_dir)
        else:
            pass
