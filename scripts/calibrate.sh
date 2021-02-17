#!/bin/bash

colmap feature_extractor \
  --database_path "$1"/database.db \
  --image_path "$1"/images \
  --ImageReader.camera_model OPENCV \
  --ImageReader.single_camera_per_folder 1 \
  --SiftExtraction.use_gpu 1 \
  --SiftExtraction.gpu_index 0 \
  | tee -a "$1"/colmap_log.txt

colmap exhaustive_matcher \
  --database_path "$1"/database.db \
  --SiftMatching.use_gpu 1 \
  --SiftMatching.gpu_index 0 \
  --SiftMatching.multiple_models 0 \
  | tee -a "$1"/colmap_log.txt

mkdir "$1"/sparse

colmap mapper \
  --database_path "$1"/database.db \
  --image_path "$1"/images \
  --output_path "$1"/sparse \
  --Mapper.multiple_models 0 \
  | tee -a "$1"/colmap_log.txt

mkdir "$1"/sparse/aligned

colmap model_aligner \
  --input_path "$1"/sparse/0 \
  --ref_images_path "$1"/ref_images.txt \
  --output_path "$1"/sparse/aligned \
  --robust_alignment_max_error 0.2 \
  | tee -a "$1"/colmap_log.txt

colmap model_converter \
  --input_path "$1"/sparse/aligned \
  --output_path "$1" \
  --output_type TXT \
  | tee -a "$1"/colmap_log.txt
