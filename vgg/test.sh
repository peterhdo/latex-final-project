for outs in 50 100 150 200 250 500
do
	echo "Testing model with $outs"

	for k in 1 2 3 4 5
	do
		echo "Testing with top k of: $k"

		#/home/ubuntu/anaconda3/envs/pytorch_latest_p36/bin/python3.6  vgg16.py \
		#       	--test-model  --num-classes $outs --dataset-path "../datasets$outs" \
		#       	--load-model-file "./latex_vgg16bn_$outs.pt" --top_k $k 
	done
done

for outs in 959 
do
	echo "Testing model with $outs"

	for k in 1 2 3 4 5
	do
		echo "Testing with top k of: $k"

		/home/ubuntu/anaconda3/envs/pytorch_latest_p36/bin/python3.6  vgg16.py \
		       	--test-model  --num-classes $outs --dataset-path "../datasets" \
		       	--load-model-file "./latex_vgg16bn_$outs.pt" --top_k $k 
	done
done


