# Pose Training

End-to-end pipeline for training custom pose estimation models.

## Capabilities

- **Converters**: CVAT XML, Lightning Pose, COCO keypoints/points/localizer formats
- **Training**: YOLO pose, POLO point-detection, PyTorch localizer heatmap models
- **Inference**: Evaluate trained models on video, export to DataFrames
- **Prep**: Dataset splitting, label filtering/simplification, data.yaml generation

::: mosaic.tracking.pose_training
    options:
      show_source: true
