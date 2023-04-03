TransFusionDetector(
  (pts_voxel_layer): Voxelization(voxel_size=[0.075, 0.075, 0.2], point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0], max_num_points=10, max_voxels=(120000, 160000))
  (pts_voxel_encoder): HardSimpleVFE()
  (pts_middle_encoder): SparseEncoder(
    (conv_input): SparseSequential(
      (0): SubMConv3d()
      (1): BatchNorm1d(16, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
      (2): ReLU(inplace=True)
    )
    (encoder_layers): SparseSequential(
      (encoder_layer1): SparseSequential(
        (0): SparseBasicBlock(
          (conv1): SubMConv3d()
          (bn1): BatchNorm1d(16, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
          (conv2): SubMConv3d()
          (bn2): BatchNorm1d(16, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (1): SparseBasicBlock(
          (conv1): SubMConv3d()
          (bn1): BatchNorm1d(16, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
          (conv2): SubMConv3d()
          (bn2): BatchNorm1d(16, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (2): SparseSequential(
          (0): SparseConv3d()
          (1): BatchNorm1d(32, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
          (2): ReLU(inplace=True)
        )
      )
      (encoder_layer2): SparseSequential(
        (0): SparseBasicBlock(
          (conv1): SubMConv3d()
          (bn1): BatchNorm1d(32, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
          (conv2): SubMConv3d()
          (bn2): BatchNorm1d(32, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (1): SparseBasicBlock(
          (conv1): SubMConv3d()
          (bn1): BatchNorm1d(32, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
          (conv2): SubMConv3d()
          (bn2): BatchNorm1d(32, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (2): SparseSequential(
          (0): SparseConv3d()
          (1): BatchNorm1d(64, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
          (2): ReLU(inplace=True)
        )
      )
      (encoder_layer3): SparseSequential(
        (0): SparseBasicBlock(
          (conv1): SubMConv3d()
          (bn1): BatchNorm1d(64, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
          (conv2): SubMConv3d()
          (bn2): BatchNorm1d(64, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (1): SparseBasicBlock(
          (conv1): SubMConv3d()
          (bn1): BatchNorm1d(64, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
          (conv2): SubMConv3d()
          (bn2): BatchNorm1d(64, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (2): SparseSequential(
          (0): SparseConv3d()
          (1): BatchNorm1d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
          (2): ReLU(inplace=True)
        )
      )
      (encoder_layer4): SparseSequential(
        (0): SparseBasicBlock(
          (conv1): SubMConv3d()
          (bn1): BatchNorm1d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
          (conv2): SubMConv3d()
          (bn2): BatchNorm1d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (1): SparseBasicBlock(
          (conv1): SubMConv3d()
          (bn1): BatchNorm1d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
          (conv2): SubMConv3d()
          (bn2): BatchNorm1d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
      )
    )
    (conv_out): SparseSequential(
      (0): SparseConv3d()
      (1): BatchNorm1d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
      (2): ReLU(inplace=True)
    )
  )
  (pts_backbone): SECOND(
    (blocks): ModuleList(
      (0): Sequential(
        (0): Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (1): BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
        (3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (4): BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        (5): ReLU(inplace=True)
        (6): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (7): BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        (8): ReLU(inplace=True)
        (9): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (10): BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        (11): ReLU(inplace=True)
        (12): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (13): BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        (14): ReLU(inplace=True)
        (15): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (16): BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        (17): ReLU(inplace=True)
      )
      (1): Sequential(
        (0): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (1): BatchNorm2d(256, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
        (3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (4): BatchNorm2d(256, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        (5): ReLU(inplace=True)
        (6): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (7): BatchNorm2d(256, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        (8): ReLU(inplace=True)
        (9): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (10): BatchNorm2d(256, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        (11): ReLU(inplace=True)
        (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (13): BatchNorm2d(256, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        (14): ReLU(inplace=True)
        (15): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (16): BatchNorm2d(256, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        (17): ReLU(inplace=True)
      )
    )
  )
  (pts_neck): SECONDFPN(
    (deblocks): ModuleList(
      (0): Sequential(
        (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(256, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
      )
      (1): Sequential(
        (0): ConvTranspose2d(256, 256, kernel_size=(2, 2), stride=(2, 2), bias=False)
        (1): BatchNorm2d(256, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
      )
    )
  )
  (pts_bbox_head): TransFusionHead(
    (loss_cls): FocalLoss()
    (loss_bbox): L1Loss()
    (loss_iou): VarifocalLoss()
    (loss_heatmap): GaussianFocalLoss()
    (shared_conv): Conv2d(512, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (heatmap_head): Sequential(
      (0): ConvModule(
        (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (activate): ReLU(inplace=True)
      )
      (1): Conv2d(128, 10, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    )
    (class_encoding): Conv1d(10, 128, kernel_size=(1,), stride=(1,))
    (decoder): ModuleList(
      (0): TransformerDecoderLayer(
        (self_attn): MultiheadAttention(
          (out_proj): Linear(in_features=128, out_features=128, bias=True)
        )
        (multihead_attn): MultiheadAttention(
          (out_proj): Linear(in_features=128, out_features=128, bias=True)
        )
        (linear1): Linear(in_features=128, out_features=256, bias=True)
        (dropout): Dropout(p=0.1, inplace=False)
        (linear2): Linear(in_features=256, out_features=128, bias=True)
        (norm1): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
        (norm2): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
        (norm3): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
        (dropout1): Dropout(p=0.1, inplace=False)
        (dropout2): Dropout(p=0.1, inplace=False)
        (dropout3): Dropout(p=0.1, inplace=False)
        (self_posembed): PositionEmbeddingLearned(
          (position_embedding_head): Sequential(
            (0): Conv1d(2, 128, kernel_size=(1,), stride=(1,))
            (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU(inplace=True)
            (3): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
          )
        )
        (cross_posembed): PositionEmbeddingLearned(
          (position_embedding_head): Sequential(
            (0): Conv1d(2, 128, kernel_size=(1,), stride=(1,))
            (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU(inplace=True)
            (3): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
          )
        )
      )
    )
    (prediction_heads): ModuleList(
      (0): FFN(
        (center): Sequential(
          (0): ConvModule(
            (conv): Conv1d(128, 64, kernel_size=(1,), stride=(1,), bias=False)
            (bn): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): ReLU(inplace=True)
          )
          (1): Conv1d(64, 2, kernel_size=(1,), stride=(1,))
        )
        (height): Sequential(
          (0): ConvModule(
            (conv): Conv1d(128, 64, kernel_size=(1,), stride=(1,), bias=False)
            (bn): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): ReLU(inplace=True)
          )
          (1): Conv1d(64, 1, kernel_size=(1,), stride=(1,))
        )
        (dim): Sequential(
          (0): ConvModule(
            (conv): Conv1d(128, 64, kernel_size=(1,), stride=(1,), bias=False)
            (bn): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): ReLU(inplace=True)
          )
          (1): Conv1d(64, 3, kernel_size=(1,), stride=(1,))
        )
        (rot): Sequential(
          (0): ConvModule(
            (conv): Conv1d(128, 64, kernel_size=(1,), stride=(1,), bias=False)
            (bn): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): ReLU(inplace=True)
          )
          (1): Conv1d(64, 2, kernel_size=(1,), stride=(1,))
        )
        (vel): Sequential(
          (0): ConvModule(
            (conv): Conv1d(128, 64, kernel_size=(1,), stride=(1,), bias=False)
            (bn): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): ReLU(inplace=True)
          )
          (1): Conv1d(64, 2, kernel_size=(1,), stride=(1,))
        )
        (heatmap): Sequential(
          (0): ConvModule(
            (conv): Conv1d(128, 64, kernel_size=(1,), stride=(1,), bias=False)
            (bn): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): ReLU(inplace=True)
          )
          (1): Conv1d(64, 10, kernel_size=(1,), stride=(1,))
        )
      )
    )
  )
)