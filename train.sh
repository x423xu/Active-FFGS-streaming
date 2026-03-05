CUDA_VISIBLE_DEVICES=8 python -m src.main +experiment=re10k \
                                    dataset.roots=[/data0/xxy/data/re10k]\
                                    data_loader.train.batch_size=2 \
                                    model.encoder.name=efficient_encoder \
                                    dataset.test_chunk_interval=10 \
                                    trainer.max_steps=300000 \
                                    model.encoder.upsample_factor=4 \
                                    model.encoder.lowest_feature_resolution=4 \
                                    model.encoder.gs_cube=false \
                                    model.encoder.monodepth_vit_type=vits \
                                    output_dir=checkpoints/efficient_encoder_depth_distill \
                                    dataset.image_shape=[252,252] \
                                    train_controller.depth_distillation=true \
                                    dataset.near=0.01 \

CUDA_VISIBLE_DEVICES=9 python -m src.main +experiment=re10k \
                                    dataset.roots=[/data0/xxy/data/re10k]\
                                    data_loader.train.batch_size=2 \
                                    model.encoder.name=efficient_encoder \
                                    dataset.test_chunk_interval=10 \
                                    trainer.max_steps=300000 \
                                    model.encoder.upsample_factor=4 \
                                    model.encoder.lowest_feature_resolution=4 \
                                    model.encoder.gs_cube=false \
                                    model.encoder.monodepth_vit_type=vits \
                                    output_dir=checkpoints/efficient_encoder_ddss_max03 \
                                    dataset.image_shape=[252,252] \
                                    train_controller.depth_distillation=true \
                                    dataset.near=0.01 \
                                    model.encoder.gaussian_adapter.gaussian_scale_max=0.3 \

CUDA_VISIBLE_DEVICES=7 python -m src.main +experiment=re10k \
                                    dataset.roots=[/data0/xxy/data/re10k]\
                                    data_loader.train.batch_size=2 \
                                    model.encoder.name=efficient_encoder \
                                    dataset.test_chunk_interval=10 \
                                    trainer.max_steps=300000 \
                                    model.encoder.upsample_factor=4 \
                                    model.encoder.lowest_feature_resolution=4 \
                                    model.encoder.gs_cube=false \
                                    model.encoder.monodepth_vit_type=vits \
                                    output_dir=checkpoints/efficient_encoder_ddvggt_max03 \
                                    dataset.image_shape=[252,252] \
                                    train_controller.depth_distillation=true \
                                    dataset.near=0.01 \
                                    model.encoder.gaussian_adapter.gaussian_scale_max=0.3 \
                                    train_controller.teacher_depth=vggt

CUDA_VISIBLE_DEVICES=8 python -m src.main +experiment=re10k \
                                    dataset.roots=[/data0/xxy/data/re10k]\
                                    data_loader.train.batch_size=2 \
                                    model.encoder.name=efficient_encoder \
                                    dataset.test_chunk_interval=10 \
                                    trainer.max_steps=300000 \
                                    model.encoder.upsample_factor=4 \
                                    model.encoder.lowest_feature_resolution=4 \
                                    model.encoder.gs_cube=false \
                                    model.encoder.monodepth_vit_type=vits \
                                    output_dir=checkpoints/efficient_encoder_ddvggt_max03 \
                                    dataset.image_shape=[252,252] \
                                    train_controller.depth_distillation=false \
                                    dataset.near=0.01 \
                                    model.encoder.gaussian_adapter.gaussian_scale_max=0.3 \
                                    train_controller.embedding_type=plucker_ray

CUDA_VISIBLE_DEVICES=9 python -m src.main +experiment=re10k \
                                    dataset.roots=[/data0/xxy/data/re10k]\
                                    data_loader.train.batch_size=2 \
                                    model.encoder.name=efficient_encoder \
                                    dataset.test_chunk_interval=10 \
                                    trainer.max_steps=300000 \
                                    model.encoder.upsample_factor=4 \
                                    model.encoder.lowest_feature_resolution=4 \
                                    model.encoder.gs_cube=false \
                                    model.encoder.monodepth_vit_type=vits \
                                    output_dir=checkpoints/ddvggt_embed_max03_near001 \
                                    dataset.image_shape=[252,252] \
                                    train_controller.depth_distillation=true \
                                    dataset.near=0.01 \
                                    model.encoder.gaussian_adapter.gaussian_scale_max=0.3 \
                                    train_controller.teacher_depth=vggt \
                                    train_controller.embedding_type=plucker_ray \
                                    train_controller.vggt_meta=true

CUDA_VISIBLE_DEVICES=9 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True conda run -p /data0/xxy/conda_envs/depthsplat --no-capture-output python -m src.main \
    +experiment=re10k \
    mode=train \
    model/encoder=mono_model \
    dataset.roots=[/data0/xxy/data/re10k] \
    dataset.image_shape=[256,256] \
    dataset.view_sampler.num_context_views=2 \
    checkpointing.pretrained_model=/data0/xxy/code/MonoSplat/outputs/2026-02-10/13-21-17/checkpoints/epoch_9-step_300000.ckpt \
    checkpointing.no_strict_load=true \
    model.encoder.enable_voxelization=true \
    model.encoder.voxel_compute_2d_branch=false \
    model.encoder.voxelization_downsample_factor=1 \
    model.encoder.voxel_conf_threshold=0.0 \
    model.encoder.voxel_max_points_per_batch=140000 \
    model.encoder.use_plucker_embedding=false \
    model.encoder.profile_voxelization=false \
    model.encoder.voxel_low_vram_arch=false \
    model.encoder.voxel_train_depth_predictor=true \
    trainer.max_steps=50 \
    trainer.num_sanity_val_steps=0 \
    data_loader.train.batch_size=2 \
    data_loader.train.num_workers=8 \
    data_loader.train.persistent_workers=false \
    ++wandb.mode=disabled

CUDA_VISIBLE_DEVICES=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True conda run -p /data0/xxy/conda_envs/depthsplat --no-capture-output python -m src.main \
    +experiment=re10k \
    mode=train \
    model/encoder=mono_model \
    dataset.roots=[/data0/xxy/data/re10k] \
    data_loader.train.batch_size=2 \
    dataset.min_views=2 \
    dataset.max_views=2 \
    model.encoder.enable_voxelization=true \
    model.encoder.use_plucker_embedding=true \
    model.encoder.voxel_train_depth_predictor=true \
    model.encoder.voxel_conf_threshold=0.0 \
    model.encoder.voxel_max_points_per_batch=140000 \
    model.encoder.voxel_low_vram_arch=true \
    model.encoder.voxelization_downsample_factor=1 \
    model.encoder.profile_voxelization=false \
    train.video_viz_interval_steps=1000 \
    trainer.max_steps=100000 \
    output_dir=checkpoints/mono_voxel

