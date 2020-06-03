./vgg16.py --save-model --epochs 10 --num-classes 100 --dataset-path '../datasets100' | tee vgg16_100classes.txt
mv latex_vgg16bn.pt latex_vgg16bn_100.pt
./vgg16.py --save-model --epochs 10 --num-classes 150 --dataset-path '../datasets150' | tee vgg16_150classes.txt
mv latex_vgg16bn.pt latex_vgg16bn_150.pt
./vgg16.py --save-model --epochs 10 --num-classes 50 --dataset-path '../datasets50' | tee vgg16_50classes.txt
mv latex_vgg16bn.pt latex_vgg16bn_50.pt
