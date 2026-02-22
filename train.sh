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