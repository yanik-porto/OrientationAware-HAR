# OrientationAware-HAR

This is the repository that contains source code for the paper "Cross-Domain Human Action Recognition from Multiview Motion and Textual Descriptions"

The paper is accepted to ICPR 2026.

## Abstract
> Robustness to domain changes is a key capability for effective deployment of human action recognition systems in real-world scenarios, where action categories at inference can present important domain shifts or even unseen actions from training. In this context, improving the recognition capabilities of Zero-Shot Action Recognition models (ZSAR), without requiring strong annotation efforts, remains a central challenge. Most ZSAR approaches assume that actions are observed under geometric conditions similar to those seen during training. In practice, variations in human body orientation and camera viewpoint add a significant domain gap in ZSAR, substantially limiting generalization to novel action?motion combinations. In this context, this paper presents a novel orientation-aware action recognition approach with improved cross-domain capabilities. Our approach combines motion cues of multiple camera viewpoints and text descriptions of human actions in the training phase. We present a new orientation-aware motion encoding network to learn different motion features, and adapt a specific orientation-aware text prompt to match the corresponding features at inference. Extensive experiments demonstrate that the proposed method consistently improves ZSAR performance across different recognition benchmarks, outperforming recent state-of-the-art zero-shot approaches on NTU-RGB+D, BABEL, NW-UCLA, and on two surveillance datasets. In addition, the learned representations exhibit strong transfer learning capabilities, yielding competitive performance on both cross-domain and same-domain recognition of seen actions.

## Installation

```shell
pip install -r requirements.txt
```

## Test

To test SAME_DOMAIN or ZSL, run the test with the config and checkpoint from the same folder.

```shell
python test.py checkpoints/ntu60/CROSS_DOMAIN/config.yaml checkpoints/ntu60/CROSS_DOMAIN/zsl.pth
```

To test ZSCD, run the chosen config from a CROSS_DOMAIN folder with the checkpoint from a SAME_DOMAIN folder.

```shell
python test.py checkpoints/ntu60/CROSS_DOMAIN/config.yaml checkpoints/babel_120/SAME_DOMAIN/best.pth
```

## Test Multi View Inference
```shell
python test_multi_view.py checkpoints/ntu60/CROSS_DOMAIN/config.yaml checkpoints/ntu60/CROSS_DOMAIN/zsl.pth --name=testset_mv
```

# Website License
<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.
