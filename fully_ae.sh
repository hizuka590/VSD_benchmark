# sh fully_ae.sh > logs/err.071546 2>&1 &
echo "Please running this file in the root of project <VSD_benchmark>"
pwd
export PYTHONPATH="./":PYTHONPATH
export path='.'
export CUDA_VISIBLE_DEVICES="0"

#python imenh/main.py \
#  --yaml_file="options/train/demop/demop_gopro-v1.yaml" \
#  --log_dir="log/train/demop-gopro-v1/" \
#  --RESUME_PATH="log/train/demop-gopro-v1/checkpoint.pth.tar" \
#  --alsologtostderr=True

python dataset/dataloader.py