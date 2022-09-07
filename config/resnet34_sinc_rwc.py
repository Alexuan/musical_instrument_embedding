# status settings
prefix = 'rwc_fil_midi_aug_asm_s100_pretrained'
description = 'stitch: false'
engine = 'vanilla'
is_train = True
gpu_ids = [0]
seed = 100
pretrain_model_pth = ''
num_instrument = 59,
num_instrument_fml = 45,
# model settings
model = dict(
    type='instr_emd_sinc',
    transform=dict(
        type='SincConv',
        out_channels=122,
        kernel_size=50,
        stride=12,
        in_channels=1,
        padding='same',
        init_type='midi',
        min_low_hz=5,
        min_band_hz=5,
        requires_grad=False,
    ),
    backbone=dict(
        type='resnet34',
        channels=[1024, 1024, 1024, 1024, 3072],
        input_size=80,
        pretrained='',
    ),
    neck=dict(
        type='LDE',
        D=32,
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
    ),
)
criterion = dict(
    type='AngleLoss'
)
loss_weight = dict(
    instr_sym=1.0,
    instr_no=1.0
)
# dataset settings
img_norm_cfg = dict(
    mean=[125.307, 122.961, 113.8575],
    std=[51.5865, 50.847, 51.255],
    to_rgb=True)
train_pipeline = [
    dict(type='CenterCrop', size=(128, 128))
]

feat_type='wav'
data_train = dict(
    dataset_mode='rwc',
    dataroot='data/RWC_MDB_I/train',
    datafile='data/RWC_MDB_I/meta/RWC-MDB-I-2001_all_inst-e.csv',
    feat_type=feat_type,
    is_train=True,
    is_stitch=False,
    num_instr=num_instrument,
    num_instr_fml=num_instrument_fml,
    batch_size=64,
    num_threads=16,
    sr=44100,
    in_memory=True,
    one_hot_all=False,
    one_hot_instr=False,
    one_hot_pitch=True,
    one_hot_velocity=False,
    one_hot_instr_src=False,
    one_hot_instr_family=False,
    encode_cat=True,
    shuffle=True,
    is_augment=False,
)
data_valid = dict(
    dataset_mode='rwc',
    dataroot='data/RWC_MDB_I/valid',
    datafile='data/RWC_MDB_I/meta/RWC-MDB-I-2001_all_inst-e.csv',
    feat_type=feat_type,
    is_train=False,
    is_stitch=False,
    num_instr=num_instrument,
    num_instr_fml=num_instrument_fml,
    batch_size=32,
    num_threads=4,
    sr=16000,
    in_memory=True,
    one_hot_all=False,
    one_hot_instr=False,
    one_hot_pitch=True,
    one_hot_velocity=False,
    one_hot_instr_src=False,
    one_hot_instr_family=False,
    encode_cat=True,
    shuffle=False,
    is_augment=False,
)
data_test = dict(
    dataset_mode='rwc',
    dataroot='data/RWC_MDB_I/test',
    datafile='data/RWC_MDB_I/meta/RWC-MDB-I-2001_all_inst-e.csv',
    feat_type=feat_type,
    is_train=False,
    is_stitch=False,
    num_instr=num_instrument,
    num_instr_fml=num_instrument_fml,
    batch_size=128,
    num_threads=4,
    sr=16000,
    in_memory=True,
    one_hot_all=False,
    one_hot_instr=False,
    one_hot_pitch=True,
    one_hot_velocity=False,
    one_hot_instr_src=False,
    one_hot_instr_family=False,
    encode_cat=True,
    shuffle=False,
    is_augment=False,
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
        n_warmup_steps=800,
        base_lr=1e-6,
        max_lr=1e-3,
        step_size_up=800,
        step_size_down=2400,
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
    dir='data/RWC_MDB_I/test'
    )
