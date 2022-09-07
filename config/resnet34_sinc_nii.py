# status settings
prefix = 'nii_sinc_midi_aug_l8_asm_instr'
description = 'sincnet: True, initialization: MIDI, Augmentation: True, Classifier: A-Softmax, LDE: L=8,'
engine = 'vanilla'
is_train = True
gpu_ids = [0]
seed = 100
pretrain_model_pth = ''
sr=24000,
num_instrument = 15,
num_instrument_fml = 15,
# model settings
model = dict(
    type='instr_emd_sinc',
    transform=dict(
        type='SincConv',
        sr=sr,
        out_channels=122,
        kernel_size=50,
        stride=12,
        in_channels=1,
        padding='same',
        init_type='midi',
        min_low_hz=5,
        min_band_hz=5,
        requires_grad=True,
    ),
    backbone=dict(
        type='resnet34',
        pretrained='log/nsynth_sinc_midi_aug_l8_asm_s100/8_77.h5',
    ),
    neck=dict(
        type='LDE',
        D=8,
        pooling='mean', 
        network_type='lde', 
        distance_type='sqr'
    ),
    head1=dict(
        type='AngularClsHead',
        num_classes=num_instrument,
        hidden_dim=512,
        m=2,
    ),
    head2=dict(
        type='AngularClsHead',
        num_classes=num_instrument_fml,
        hidden_dim=512,
        m=2,
    )
)
criterion = dict(
    type='AngleLoss'
)
loss_weight = dict(
    label_A=1,
    label_B=0,
)
# dataset settings
feat_type='wav'
data_train = dict(
    dataset_mode='nii',
    dataroot='data/NII-URMP',
    datafile='scp_seg_200f/train.lst.validated.seg',
    feat_type=feat_type,
    is_train=True,
    num_instr=num_instrument,
    num_instr_fml=num_instrument_fml,
    batch_size=32,
    num_threads=10,
    sr=sr,
    in_memory=True,
    one_hot_all=False,
    one_hot_instr=False,
    one_hot_pitch=True,
    one_hot_velocity=False,
    one_hot_instr_src=False,
    one_hot_instr_family=False,
    encode_cat=True,
    shuffle=True,
    is_augment=True,
    augment=dict(
        type='stitch',
        min_sec=3,
        max_sec=5,
    )
)
data_valid = dict(
    dataset_mode='nii',
    dataroot='data/NII-URMP',
    datafile='scp_seg_200f/val.lst.validated.seg',
    feat_type=feat_type,
    is_train=True,
    num_instr=num_instrument,
    num_instr_fml=num_instrument_fml,
    batch_size=128,
    num_threads=10,
    sr=sr,
    in_memory=True,
    one_hot_all=False,
    one_hot_instr=False,
    one_hot_pitch=True,
    one_hot_velocity=False,
    one_hot_instr_src=False,
    one_hot_instr_family=False,
    encode_cat=True,
    shuffle=True,
    is_augment=True,
    augment=dict(
        type='stitch',
        min_sec=3,
        max_sec=5,
    )
)
data_test = dict(
    dataset_mode='nii',
    dataroot='data/NII-URMP',
    datafile='scp_seg_200f/test.lst.validated.seg',
    feat_type=feat_type,
    is_train=True,
    num_instr=num_instrument,
    num_instr_fml=num_instrument_fml,
    batch_size=128,
    num_threads=10,
    sr=sr,
    in_memory=True,
    one_hot_all=False,
    one_hot_instr=False,
    one_hot_pitch=True,
    one_hot_velocity=False,
    one_hot_instr_src=False,
    one_hot_instr_family=False,
    encode_cat=True,
    shuffle=True,
    is_augment=True,
    augment=dict(
        type='stitch',
        min_sec=3,
        max_sec=5,
    )
)
# optimizer
optimizer = dict(
    type='Adam',
    lr=1e-4,
    weight_decay=1e-5,
    betas=(0.9, 0.98), 
    eps=1e-09, 
    amsgrad=True,
    grad_clip=None,
    lr_scheduler=dict(
        type='ScheduledOptim',
        n_warmup_steps=1000,
        base_lr=1e-6,
        max_lr=1e-3,
        step_size_up=8000,
        step_size_down=24000,
        mode='exp_range',
    )
)
# learning policy
lr_config = dict(policy='step', step=[30, 60, 90])
lr_weight = dict(type='loss_weight', loss_cls=1., loss_pl=3.)
runner = dict(type='EpochBasedRunner', max_epochs=30)
# log 
log_config = dict(
    interval=1,
    log_level = 'INFO',
    dir='log/'+prefix,
    hooks=[
        dict(type='TextLoggerHook'),
    ])
# yapf:enable
dist_params = dict(backend='nccl')
workflow = [('train', 1)]

decode = dict(
    dir='data/nii/nii-test'
)