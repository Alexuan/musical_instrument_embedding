# musical_instrument_embedding

Thanks for your interests in our musical instrument embedding work "[Use of Speaker Recognition Approaches for Learning and Evaluating Embedding Representations of Musical Instrument Sounds](https://ieeexplore.ieee.org/document/9670718)" in IEEE/ACM Transactions on Audio, Speech, and Language Processing!! 
The code is continuely tidying up.  The current version still has redundant part for development.  Feel free to contact me（<xuanshi@usc.edu>) if you have any problems.  I generally reply emails in 24 hours.

Train:
```
python train.py config/resnet34_sinc_nsynth.py
```

Inference:
```
python inference.py --config_file=config/resnet34_sinc_nsynth.py --pretrain_model_pth='log/nsynth_sinc_midi_aug_l8_asm_s100/8_77.h5'
```
EER: 0.02935

