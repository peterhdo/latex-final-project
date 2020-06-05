/home/ubuntu/anaconda3/envs/pytorch_latest_p36/bin/python3.6  vgg16.py --save-model --epochs 10 --num-classes 100 --dataset-path '../datasets100' | tee vgg16_100classes.txt
mv latex_vgg16bn.pt latex_vgg16bn_100.pt
# Have to use this explictly as it fails in tmux / screen (the paths of which python I'm using)
/home/ubuntu/anaconda3/envs/pytorch_latest_p36/bin/python3.6 vgg16.py --save-model --epochs 10 --num-classes 150 --dataset-path '../datasets150' | tee vgg16_150classes.txt
mv latex_vgg16bn.pt latex_vgg16bn_150.pt
/home/ubuntu/anaconda3/envs/pytorch_latest_p36/bin/python3.6 vgg16.py --save-model --epochs 10 --num-classes 50 --dataset-path '../datasets50' | tee vgg16_50classes.txt
mv latex_vgg16bn.pt latex_vgg16bn_50.pt


