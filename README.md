# musical_instrument_embedding

Thanks for your interests in our musical instrument embedding work!!  Apologies on the delay.  Feel free to contact me（<xuanshi@usc.edu>) if you have any problems.  I generally reply emails in 24 hours.

The code is continuely tidying up.  The current version still has redundant part for development.   

Train:
```
python train.py config/resnet34_sinc_nsynth.py
```

Inference:
```
python inference.py --config_file=config/resnet34_sinc_nsynth.py --pretrain_model_pth='log/nsynth_sinc_midi_aug_l8_asm_s100/8_77.h5'
```
EER: 0.02935

