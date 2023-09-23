## Fine-grained Human Body Motion Generation Based On Conditional Variational Autoencoder


Code for "Fine-grained Human Body Motion Generation Based On Conditional Variational Autoencoder"
### Dependencies
* Python >= 3.8
* [PyTorch](https://pytorch.org) >= 1.8
* Tensorboard
* numba
tested on pytorch == 1.10.1

Human body model and data [link](https://drive.google.com/drive/folders/1CD5WdsOHBk5btt7Ilr-J31keNgLKIDG7)

### Training and Evaluation
* We provide YAML configs inside ``motion_pred/cfg``: `[dataset]_rnn.yml` and `[dataset]_act_classifier.yml` for the main model and the classifier (for evaluation) respectively. These configs correspond to pretrained models inside ``results``.
#### Train
python generation_exp_vae_act_babel.py --gpu_index 0 --cfg ours_rnn  --seed 0
#### Test
python generation_eval_vae_act_render_video.py  --gpu_index 0  --cfg "new_action"  --iter 5000 --seed 0
#### Evaluate
python generation_eval_ours.py --iter 10000 --class_iter 1000 --nk 10 --bs 8 --num_samp 50 --num_seed 1 --stop_fn 5 --cfg "new_action" --cfg_classifier new_action_class --gpu_index 0 --threshold 0.025

Paper Link: https://drive.google.com/file/d/1hGD3mBuKunKsV3VDlWkLAxJyu3V8idI2/view?usp=sharing
