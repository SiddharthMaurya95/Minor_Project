### Consolidated Efficiency Results

| Model Architecture | Trainable Parameters | Model Size | MACs | FLOPs | CPU Latency | GPU Latency | Accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PhySSL-Net | 46,294 | 0.18 MB | 9.07M | 18.15M | 1.324 ms | 2.646 ms | 87.21% |
| Vanilla CNN | 65,314 | 0.25 MB | 12.80M | 25.60M | 0.710 ms | 0.862 ms | 85.26% |
| ConvNeXt-Femto | 109,890 | 0.42 MB | 21.54M | 43.08M | 3.639 ms | 5.941 ms | 86.54% |
| ResNet50V2Small | 751,522 | 2.87 MB | 147.30M | 294.60M | 5.555 ms | 11.604 ms | 84.23% |
| VGG16Small | 894,498 | 3.41 MB | 175.32M | 350.64M | 1.849 ms | 2.326 ms | 86.96% |