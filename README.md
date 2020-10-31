# FFN_mask

EM_mask is a Python library for large scale tissue mask predictions on serial EM data inspired by and compatible with FFN standard (https://github.com/google/ffn). It can be used to run UNets on an arbiturarily large precomputed volume chunkwise with MPI parallelization. It is built-in with tensorflow implementations of [classic 2D/3D UNets](https://arxiv.org/abs/1505.04597), and [Distance Transformed UNets](https://arxiv.org/pdf/1805.02718.pdf). 

## Installation
```bash
pip install -e .
```

## Usage

1. Prepare training data into an h5 file with grayscale image and binary mask label(soma, vessicle cloud, synaptic junctions) in z,y,x shape

2. Prepare training coordinates with the same technique in (https://github.com/google/ffn) by evenly sample coordinates depending on coverage percentage of mask. 

3. Training

Assuming a dual-gpu setup:
```bash
horovodrun -n 2 -H localhost:2 \
    python {$INSTALL_DIR}/train.py \
    --data_volumes=$INPUT_NAME:$INPUT_PATH:image \
    --label_volumes=$INPUT_NAME:$INPUT_PATH:label \
    --tf_coords=$TF_COORD_FILES \
    --train_dir=$CHECKPOINT_DIR \
    --model_name='models.unets.unet_dtu_2_pad_concat' \
    --model_args="{\"fov_size\": [128, 128, 12], \"num_classes\": 1, \"label_size\": [128, 128, 12]}" \
    --learning_rate=0.001 \
    --batch_size=2 \
    --image_mean=120 \
    --image_stddev=46 \
    --rotation \
    --max_steps 100000 \
```

4. Inference

Inference can be performed on either h5 data or precomputed, refer to [Neuroglancer](https://github.com/google/neuroglancer), [CloudVolume](https://github.com/seung-lab/cloud-volume)
```bash
mpirun -n 2 \
  python {$INSTALL_DIR}/predict_precomputed.py \
    --input_volume=$INPUT_PRECOMPUTED_DIR \
    --input_offset='0,0,0' \
    --input_size='512,512,128' \
    --input_mip=1 \
    --output_volume=$OUTPUT_PRECOMPUTED_DIR \
    --model_name='models.unets.unet_dtu_2_pad_concat' \
    --model_args="{\"fov_size\": [218, 218, 23], \"num_classes\": 1}" \
    --model_checkpoint=$CHECKPOINT \
    --overlap='32,32,16' \
    --batch_size=2 \
    --image_mean=120 \
    --image_stddev=46 \
    --var_threshold=10 \
    --use_gpu=0,1 \
    --alsologtostderr
```
The output will be two precomputed volumes "class_label" and "logits"

## License
[MIT](https://choosealicense.com/licenses/mit/)