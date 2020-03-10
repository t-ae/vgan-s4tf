# VGAN on Swift for TensorFlow

- [Variational Discriminator Bottleneck: Improving Imitation Learning, Inverse RL, and GANs by Constraining Information Flow](https://arxiv.org/abs/1810.00821)
- [Official implementation](https://github.com/akanazawa/vgan)

Also used spectral normalization in discriminator.

- [Spectral Normalization for Generative Adversarial Networks](https://arxiv.org/abs/1802.05957)

## Run

```bash
$ swift run VGAN $(TRAINING_IMAGE_DIRECTORY)
```

## Dockerfile

https://github.com/t-ae/s4tf-docker

## Result

[FFHQ](https://github.com/NVlabs/ffhq-dataset) / 256x256

![random1](https://user-images.githubusercontent.com/12446914/76323901-a7022a00-6328-11ea-80ee-6b040e232a2a.png)

More results: https://github.com/t-ae/vgan-s4tf/tree/master/Results