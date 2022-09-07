# status settings
prefix = 'probe_ml_rwc_midi_style_strings'
description = 'probing information encoded in embedding: playing style'
engine = 'probe'
is_train = True
gpu_ids = [0]
seed = 100
pretrain_model_pth = ''
probe_type = 'instr_fml'
num_instrument = 1006,
num_instrument_fml = 11,
num_pitch = 112,
num_pitch_octave = 10,
num_velocity = 5,
num_quality = 10,
num_style = 20,
num_manu = 12,
num_dynam = 3,
# model settings
model = dict(
    type='SVM',
    perceptron=dict(
        type='LinearClsHead',
        num_classes=num_style,
        in_channels=512,
        hidden_dim=100,
        softmax=True,
        sigmoid=False,
    )
)
criterion = dict(
    type='NLL'
)
loss_weight = dict(
    instr=0.5,
    instr_fml=1.5
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
data = dict(
    dataset_mode='rwc_feat',
    dataroot='data/RWC_MDB_I/probing_data/feat_instr/all_test',
    datafile='data/RWC_MDB_I/meta/RWC-MDB-I-2001_all_inst-e.csv',
    feat_type=feat_type,
    is_train=True,
    num_instr=num_instrument,
    num_instr_fml=num_instrument_fml,
    batch_size=128,
    num_threads=10,
    sr=16000,
    in_memory=True,
    one_hot_all=False,
    one_hot_instr=False,
    one_hot_pitch=True,
    one_hot_pitch_octave=True,
    one_hot_velocity=False,
    one_hot_instr_src=False,
    one_hot_instr_family=False,
    one_hote_quality=True,
    encode_cat=True,
    shuffle=True,
    is_augment=False,
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
        n_warmup_steps=8000,
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
runner = dict(type='EpochBasedRunner', max_epochs=100)
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
    dir='data/nsynth/nsynth-train'
)
