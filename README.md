# Object Segmentation to Reduce Contextual Shortcut Learning

This project investigates whether automatic foreground segmentation can reduce contextual shortcut learning in image classification. Using a 10-class subset of the iNaturalist-12K dataset, foreground objects were automatically extracted using Segment Anything (SAM) and CLIP, and used to augment training data for CNN classifiers.

The experiments compared different probabilities of replacing original images with segmented versions. Partial segmentation reduced the train-validation gap, suggesting a regularizing effect, but did not consistently improve test performance. Full segmentation substantially degraded performance, likely due to imperfect masks and the removal of useful contextual information. A pretrained ResNet18 also clearly outperformed the custom CNN, highlighting the benefits of transfer learning on the relatively small dataset.

Paper:
https://github.com/robin-szy/Deep_Learning_Uni/blob/main/Report/dl-report.pdf
