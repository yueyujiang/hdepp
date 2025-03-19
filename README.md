## This is the code for the paper: [Learning Hyperbolic Embedding for Phylogenetic Tree Placement and Updates](https://www.mdpi.com/2079-7737/11/9/1256)

## Training:
### hyperbolic depp
```
python train_depp.py backbone_tree_file=$backbone_tree backbone_seq_file=$backbone_seq weighted_method=’fm’ distance_mode=’hyperbolic’ embedding_size=$embedding_size
```
### Euclidean deep
```
python train_depp.py backbone_tree_file=$backbone_tree backbone_seq_file=$backbone_seq patience=5 lr=1e-4 embedding_size=$embedding_size
```

## Query time
```
python depp_distance.py query_seq_file=$query_seq backbone_seq_file=$backbone_seq model_path=$model_path outdir=$out_dir
```

