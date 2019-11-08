# Provable Training and Verification Approaches Towards Robust Neural Networks
Recently, provable (i.e. certified) adversarial robustness and robustness verification becomes a hot research topic.

In contrast to empirical robustness and empirical adversarial attacks, (common) provable robustness neural networks give a rigorous guarantee about the successful rate bound, that **no** existing or future attacks will break. Robustness verification approaches strongly connect with provable robustness, which verifies such successful rate bound given a neural network model.

Trained on training set, the provable robustness is often measure by **robust accuracy** or **error rate** on the test set. One sample is considered accurate if and only if we can prove that there is no adversararial samples exist in the neighborhood, i.e., the model always outputs the current prediction label in the neighborhood. The neighborhood is usually defined by L-norm balls. For example, L-infty balls are $\{x + \delta: \lVert \delta \rVert_\infty \le \epsilon\}$ , L-2 balls are $\{x + \delta: \lVert \delta \rVert_2 \le \epsilon\}$. The size of the ball is controlled by eps ($\epsilon$).

Better provable robustness can be achieved by better provable robust training approaches which yield more robust models, tighter robustness verification approaches which yield tighter guarantees, or jointly.

#### Scoop of the Repo

Currently, works in literature mainly focus on image classification tasks with datasets MNIST, CIFAR10, ImageNet, FashionMNIST, and SVHN.

Mainly perturbation norm are L-2 balls and l-infty balls.

This repo mainly records recent progress of above settings, while advances in other settings are recorded in the attached paperlist.

We only consider **single model** robustness.

#### Contact & Updates

We are trying to keep track of all important advances in provable robustness, but may still miss some works.

Please feel free to contact us (Linyi(linyi2@illinois.edu) @ [UIUC Secure Learning Lab](https://aisecure.github.io/) & [Illinois ASE Group](http://asenews.blogspot.com/)) or commit your updates :)

## Main Leaderboard

### ImageNet

All input images have three channels; each pixel is in range [0, 255].

#### L2

##### eps = 0.2

| Defense                                                      | Author        | Model Structure | Accuracy |      |
| ------------------------------------------------------------ | ------------- | --------------- | -------- | ---- |
| [Certified Robustness to Adversarial Examples with Differential Privacy](https://arxiv.org/pdf/1802.03471.pdf) | Lecuyer et al | Inception V3    | 40%      |      |



##### eps = 0.5

| Defense                                                      | Author       | Model Structure | Accuracy |      |
| ------------------------------------------------------------ | ------------ | --------------- | -------- | ---- |
| [Provably Robust Deep Learning via Adversarially Trained Smoothed Classifiers](https://arxiv.org/pdf/1906.04584.pdf) | Salman et al | ResNet-50       | 56%      |      |
| [Certified Adversarial Robustness via Randomized Smoothing](https://arxiv.org/pdf/1902.02918.pdf) | Cohen et al  | ResNet-50       | 49%      |      |

All above approaches use Randomized Smoothing (Cohen et al) to derive certification, with wrong probability at 0.1%.



##### eps = 1.0

| Defense                                                      | Author       | Model Structure | Accuracy |      |
| ------------------------------------------------------------ | ------------ | --------------- | -------- | ---- |
| [Provably Robust Deep Learning via Adversarially Trained Smoothed Classifiers](https://arxiv.org/pdf/1906.04584.pdf) | Salman et al | ResNet-50       | 43%      |      |
| [Certified Adversarial Robustness via Randomized Smoothing](https://arxiv.org/pdf/1902.02918.pdf) | Cohen et al  | ResNet-50       | 37%      |      |

All above approaches use Randomized Smoothing (Cohen et al) to derive certification, with wrong probability at 0.1%.



##### eps = 2.0

| Defense                                                      | Author       | Model Structure | Accuracy |      |
| ------------------------------------------------------------ | ------------ | --------------- | -------- | ---- |
| [Provably Robust Deep Learning via Adversarially Trained Smoothed Classifiers](https://arxiv.org/pdf/1906.04584.pdf) | Salman et al | ResNet-50       | 27%      |      |
| [Certified Adversarial Robustness via Randomized Smoothing](https://arxiv.org/pdf/1902.02918.pdf) | Cohen et al  | ResNet-50       | 19%      |      |

All above approaches use Randomized Smoothing (Cohen et al) to derive certification, with wrong probability at 0.1%.



##### eps = 3.0

| Defense                                                      | Author       | Model Structure | Accuracy |      |
| ------------------------------------------------------------ | ------------ | --------------- | -------- | ---- |
| [Provably Robust Deep Learning via Adversarially Trained Smoothed Classifiers](https://arxiv.org/pdf/1906.04584.pdf) | Salman et al | ResNet-50       | 20%      |      |
| [Certified Adversarial Robustness via Randomized Smoothing](https://arxiv.org/pdf/1902.02918.pdf) | Cohen et al  | ResNet-50       | 12%      |      |

All above approaches use Randomized Smoothing (Cohen et al) to derive certification, with wrong probability at 0.1%.

#### L-Infty

##### eps=1/255

| Defense                                                      | Author       | Model Structure  | Accuracy |                                                              |
| ------------------------------------------------------------ | ------------ | ---------------- | -------- | ------------------------------------------------------------ |
| [Provably Robust Deep Learning via Adversarially Trained Smoothed Classifiers](https://arxiv.org/pdf/1906.04584.pdf) | Salman et al | ResNet-50        | 36.8%    | transformed from L-2 robustness; wrong prob. 0.001           |
| [Certified Adversarial Robustness via Randomized Smoothing](https://arxiv.org/pdf/1902.02918.pdf) | Cohen et al  | ResNet-50        | 28.6%    | transformed from L-2 robustness by Salman et al; wrong prob. 0.001 |
| [On the Effectiveness of Interval Bound Propagation for Training Verifiably Robust Models](https://arxiv.org/pdf/1810.12715.pdf) | Gowal et al  | WideResNet-10-10 | 6.13%    | Dataset downscaled to 64 x 64                                |



##### eps=1.785/255

| Defense                                                      | Author     | Model Structure | Accuracy |                                |
| ------------------------------------------------------------ | ---------- | --------------- | -------- | ------------------------------ |
| [MixTrain: Scalable Training of Verifiably Robust Neural Networks](https://arxiv.org/pdf/1811.02625.pdf) | Wang et al | ResNet          | 19.4%    |                                |
| [Scaling provable adversarial defenses](https://arxiv.org/pdf/1805.12514.pdf) | Wong et al | ResNet          | 5.1%     | Run and reported by Wang et al |

In above table, the dataset is ImageNet-200 rather than ImageNet-1000 in other tables.

### CIFAR-10

All input images have three channels; 32 x 32 x 3 size; each pixel is in range [0, 255].

#### L2

##### eps=0.14

| Defense                                                      | Author        | Model Structure | Accuracy |                                        |
| ------------------------------------------------------------ | ------------- | --------------- | -------- | -------------------------------------- |
| [Scaling provable adversarial defenses](https://arxiv.org/pdf/1805.12514.pdf) | Wong et al    | Resnet          | 51.96%   | 36/255; transformed from L-infty 2/255 |
| [Certified Robustness to Adversarial Examples with Differential Privacy](https://arxiv.org/pdf/1802.03471.pdf) | Lecuyer et al | Resnet          | 40%      |                                        |
| (Verification) [Efficient Neural Network Robustness Certification with General Activation Functions](https://papers.nips.cc/paper/7742-efficient-neural-network-robustness-certification-with-general-activation-functions.pdf) | Zhang et al   | ResNet-20       | 0%       | Reported by Cohen et al                |



##### eps=0.25

| Defense                                                      | Author       | Model Structure | Accuracy |                          |
| ------------------------------------------------------------ | ------------ | --------------- | -------- | ------------------------ |
| [Provably Robust Deep Learning via Adversarially Trained Smoothed Classifiers](https://arxiv.org/pdf/1906.04584.pdf) | Salmon et al | ResNet-110      | 82%      |                          |
| [Unlabeled Data Improves Adversarial Robustness](https://arxiv.org/pdf/1905.13736.pdf) | Carmon et al | ResNet 28-10    | 72%      | interpolated from Fig. 1 |
| [Certified Adversarial Robustness via Randomized Smoothing](https://arxiv.org/pdf/1902.02918.pdf) | Cohen et al  | ResNet-110      | 61%      |                          |
| (Verification) [Efficient Neural Network Robustness Certification with General Activation Functions](https://papers.nips.cc/paper/7742-efficient-neural-network-robustness-certification-with-general-activation-functions.pdf) | Zhang et al  | ResNet-20       | 0%       | Reported by Cohen et al  |

All above approaches use Randomized Smoothing (Cohen et al) to derive certification, with wrong probability at 0.1%.

##### eps=0.5

| Defense                                                      | Author       | Model Structure | Accuracy |                          |
| ------------------------------------------------------------ | ------------ | --------------- | -------- | ------------------------ |
| [Provably Robust Deep Learning via Adversarially Trained Smoothed Classifiers](https://arxiv.org/pdf/1906.04584.pdf) | Salmon et al | ResNet-110      | 65%      |                          |
| [Unlabeled Data Improves Adversarial Robustness](https://arxiv.org/pdf/1905.13736.pdf) | Carmon et al | ResNet 28-10    | 61%      | interpolated from Fig. 1 |
| [Certified Adversarial Robustness via Randomized Smoothing](https://arxiv.org/pdf/1902.02918.pdf]) | Cohen et al  | ResNet-110      | 43%      |                          |

All above approaches use Randomized Smoothing (Cohen et al) to derive certification, with wrong probability at 0.1%.

##### eps=1.0

| Defense                                                      | Author       | Model Structure | Accuracy |      |
| ------------------------------------------------------------ | ------------ | --------------- | -------- | ---- |
| [Provably Robust Deep Learning via Adversarially Trained Smoothed Classifiers](https://arxiv.org/pdf/1906.04584.pdf) | Salmon et al | ResNet-110      | 39%      |      |
| [Certified Adversarial Robustness via Randomized Smoothing](https://arxiv.org/pdf/1902.02918.pdf) | Cohen et al  | ResNet-110      | 22%      |      |

All above approaches use Randomized Smoothing (Cohen et al) to derive certification, with wrong probability at 0.1%.

##### eps=1.5

| Defense                                                      | Author       | Model Structure | Accuracy |      |
| ------------------------------------------------------------ | ------------ | --------------- | -------- | ---- |
| [Provably Robust Deep Learning via Adversarially Trained Smoothed Classifiers](https://arxiv.org/pdf/1906.04584.pdf) | Salmon et al | ResNet-110      | 32%      |      |
| [Certified Adversarial Robustness via Randomized Smoothing](https://arxiv.org/pdf/1902.02918.pdf) | Cohen et al  | ResNet-110      | 14%      |      |

All above approaches use Randomized Smoothing (Cohen et al) to derive certification, with wrong probability at 0.1%.

#### L Infty

##### eps=2/255

| Defense/Verification                                         | Author          | Model Structure | Accuracy    |                                                              |
| ------------------------------------------------------------ | --------------- | --------------- | ----------- | ------------------------------------------------------------ |
| [Provably Robust Deep Learning via Adversarially Trained Smoothed Classifiers](https://arxiv.org/pdf/1906.04584.pdf) | Salmon et al    | ResNet-110      | 68.2%       | transformed from L-2 robustness; wrong prob. 0.1%            |
| (Verification) [Efficient Neural Network Verification with Exactness Characterization](http://auai.org/uai2019/proceedings/papers/164.pdf) | Dvijotham et al | Small CNN       | 65.4%       |                                                              |
| [Unlabeled Data Improves Adversarial Robustness](https://arxiv.org/pdf/1905.13736.pdf) | Carmon et al    | ResNet 28-10    | 63.8% ± 0.5 | wrong prob. 0.1%                                             |
| [Robustra: Training Provable Robust Neural Networks over Reference Adversarial Space](https://www.ijcai.org/proceedings/2019/654) | Li et al        | CNN             | 56.32%      |                                                              |
| [Scaling provable adversarial defenses](https://arxiv.org/pdf/1902.02918.pdf) | Wong et al      | Resnet          | 53.89%      |                                                              |
| [Differentiable Abstract Interpretation for Provably Robust Neural Networks](http://proceedings.mlr.press/v80/mirman18b/mirman18b.pdf) | Mirman et al    | Residual        | 52.2%       | ~1.785/255, only evaluated from 500 samples among all 10,000 |
| [MixTrain: Scalable Training of Verifiably Robust Neural Networks](https://arxiv.org/pdf/1811.02625.pdf) | Wang et al      | Resnet          | 50.4%       |                                                              |
| (Verification)[Evaluating Robustness of Neural Networks with Mixed Integer Programming](https://arxiv.org/pdf/1711.07356.pdf) | Tjeng et al     | CNN             | 50.20%      | MILP verification on Wong et al. model                       |
| [On the Effectiveness of Interval Bound Propagation for Training Verifiably Robust Models](https://arxiv.org/pdf/1810.12715.pdf) | Gowal et al     | CNN             | 50.02%      |                                                              |
| [Towards Stable and Efficient Training of Verifiably Robust Neural Networks](https://arxiv.org/pdf/1906.06316.pdf) | Zhang et al   | small CNN | 47.54%   | pick the best number |
| [Training for Faster Adversarial Robustness Verification via Inducing ReLU Stability](https://arxiv.org/pdf/1809.03008.pdf) | Xiao et al      | CNN             | 45.93%      |                                                              |
| (Verification) [An Abstract Domain for Certifying Neural Networks](https://files.sri.inf.ethz.ch/website/papers/DeepPoly.pdf) | Singh et al     | CNN             | 40%         | ~2.04/255, only evaluated from 100 samples among all 10,000  |
| (Verification)[A Dual Approach to Scalable Verification of Deep Networks](http://auai.org/uai2018/proceedings/papers/204.pdf) | Dvijotham et al | CNN             | 20%         | LP-Dual verification on Uesato et al. model; interpolated from Fig. 2(a) |



##### eps=8/255

| Defense/Verification                                         | Author           | Model Structure    | Accuracy |                                                              |
| ------------------------------------------------------------ | ---------------- | ------------------ | -------- | ------------------------------------------------------------ |
| [Fast and Stable Interval Bounds Propagation for Training Verifiably Robust Models](https://arxiv.org/pdf/1906.00628.pdf) | Morawiecki et al | Small CNN          | 39.88%   |                                                              |
| [Differentiable Abstract Interpretation for Provably Robust Neural Networks](http://proceedings.mlr.press/v80/mirman18b/mirman18b.pdf) | Mirman et al     | Small CNN          | 37.4%    | ~7.65/255, only evaluated from 500 samples among all 10,000  |
| [On the Effectiveness of Interval Bound Propagation for Training Verifiably Robust Models](https://arxiv.org/pdf/1810.12715.pdf) | Gowal et al      | CNN                | 32.04%   |  Practically reproducible verified error is about 28% - 29% according to Zhang et al |
| [Towards Stable and Efficient Training of Verifiably Robust Neural Networks](https://arxiv.org/pdf/1906.06316.pdf) | Zhang et al   | large CNN | 29.21%   | pick the best number |
| [Training Verified Learners with Learned Verifiers](https://arxiv.org/pdf/1805.10265.pdf) | Dvijotham et al  | Predictor-Verifier | 26.67%   |                                                              |
| [Robustra: Training Provable Robust Neural Networks over Reference Adversarial Space](https://www.ijcai.org/proceedings/2019/654) | Li et al         | CNN                | 25.13%   |                                                              |
| [A Provable Defense for Deep Residual Networks](https://arxiv.org/pdf/1903.12519.pdf) | Mirman et al     | ResNet-Tiny        | 23.2%    |                                                              |
| [Evaluating Robustness of Neural Networks with Mixed Integer Programming](https://arxiv.org/pdf/1711.07356.pdf) | Tjeng et al      | CNN                | 22.40%   |                                                              |
| [Scaling provable adversarial defenses](https://arxiv.org/pdf/1902.02918.pdf) | Wong et al       | Resnet             | 21.78%   |                                                              |
| (Verification) [Boosting Robustness Certification of Neural Networks](https://files.sri.inf.ethz.ch/website/papers/DeepZ.pdf) | Singh et al      | Small CNN          | 21%      | ~7.65/255, only evaluated from 500 samples among all 10,000  |
| [Training for Faster Adversarial Robustness Verification via Inducing ReLU Stability](https://arxiv.org/pdf/1809.03008.pdf) | Xiao et al       | CNN                | 20.27%   |                                                              |
| (Verification) [A Dual Approach to Scalable Verification of Deep Networks](http://auai.org/uai2018/proceedings/papers/204.pdf) | Dvijotham et al  | CNN                | 0%       | LP-Dual verification on Uesato et al. model; interpolated from Fig. 2(a) |



### MNIST

All input images are grayscale; 28 x 28 size; each pixel is in range [0, 1].

#### L2

##### eps=1.58

eps=1.58 is transformed from L-infty eps=0.1.

| Defense/Verification                                         | Author     | Model Structure | Accuracy |      |
| ------------------------------------------------------------ | ---------- | --------------- | -------- | ---- |
| [Scaling provable adversarial defenses](https://arxiv.org/pdf/1902.02918.pdf) | Wong et al | Small CNN       | 88.14%   |      |



#### L-Infty

##### eps=0.1

| Defense/Verification                                         | Author            | Model Structure    | Accuracy |                                                  |
| ------------------------------------------------------------ | ----------------- | ------------------ | -------- | ------------------------------------------------ |
| [Robustra: Training Provable Robust Neural Networks over Reference Adversarial Space](https://www.ijcai.org/proceedings/2019/0654.pdf) | Li et al          | large CNN          | 97.91%   |                                                  |
| [On the Effectiveness of Interval Bound Propagation for Training Verifiably Robust Models](https://arxiv.org/pdf/1810.12715.pdf) | Gowal et al       | CNN                | 97.77%   |                                                  |
| (Verification) [Evaluating Robustness of Neural Networks with Mixed Integer Programming](https://arxiv.org/abs/1711.07356) | Tjeng et al       | large CNN          | 97.26%   |                                                  |
| [MixTrain: Scalable Training of Verifiably Robust Neural Networks](https://arxiv.org/pdf/1811.02625.pdf) | Wang et al        | small CNN          | 97.1%    |                                                  |
| (Verification) [Boosting Robustness Certification of Neural Networks](https://files.sri.inf.ethz.ch/website/papers/DeepZ.pdf) | Singh et al       | ConvSuper          | 97%      | Only evaluated from 100 samples among all 10,000 |
| [Differentiable Abstract Interpretation for Provably Robust Neural Networks](http://proceedings.mlr.press/v80/mirman18b/mirman18b.pdf) | Mirman et al      | big CNN            | 96.6%    | Only evaluated from 500 samples among all 10,000 |
| [Scaling Provable Adversarial Defenses](https://arxiv.org/pdf/1805.12514.pdf) | Wong et al        | large CNN          | 96.33%   |                                                  |
| [Towards Stable and Efficient Training of Verifiably Robust Neural Networks](https://arxiv.org/pdf/1906.06316.pdf) | Zhang et al   | small CNN | 95.79%   | pick the best number |
| [Training Verified Learners with Learned Verifiers](https://arxiv.org/pdf/1805.10265.pdf) | Dvijotham et al   | Predictor-Verifier | 95.56%   |                                                  |
| [Provable Defenses against Adversarial Examples via the Convex Outer Adversarial Polytope](https://arxiv.org/pdf/1711.00851.pdf) | Wong et al        | CNN                | 94.18%   |                                                  |
| (Verification) [Efficient Neural Network Verification with Exactness Characterization](http://auai.org/uai2019/proceedings/papers/164.pdf) | Dvijotham et al   | Grad-NN            | 83.68%   |                                                  |
| (Verification) [A Convex Relaxation Barrier to Tight Robustness Verification of Neural Networks](https://arxiv.org/pdf/1902.08722.pdf) | Salman et al      | CNN                | 82.76%   |                                                  |
| (Verification) [Semidefinite relaxations for certifying robustness to adversarial examples](https://arxiv.org/pdf/1811.01057.pdf) | Raghunathan et al | small NN           | 82%      |                                                  |
| [Certified Defenses against Adversarial Examples](https://arxiv.org/pdf/1801.09344.pdf) | Raghunathan et al | 2-layer NN         | 65%      |                                                  |



##### eps=0.3

| Defense/Verification                                         | Author       | Model Structure | Accuracy |      |
| ------------------------------------------------------------ | ------------ | --------------- | -------- | ---- |
| [Towards Stable and Efficient Training of Verifiably Robust Neural Networks](https://arxiv.org/pdf/1906.06316.pdf) | Zhang et al   | large CNN | 92.54%   | pick the best number |
| [On the Effectiveness of Interval Bound Propagation for Training Verifiably Robust Models](https://arxiv.org/pdf/1810.12715.pdf) | Gowal et al  | CNN             | 91.95%   |      |
| [Robustra: Training Provable Robust Neural Networks over Reference Adversarial Space](https://www.ijcai.org/proceedings/2019/0654.pdf) | Li et al     | small CNN       | 83.09%   |      |
| (Verification) [Evaluating Robustness of Neural Networks with Mixed Integer Programming](https://arxiv.org/abs/1711.07356) | Tjeng et al  | large CNN       | 75.81%   |      |
| [MixTrain: Scalable Training of Verifiably Robust Neural Networks](https://arxiv.org/pdf/1811.02625.pdf) | Wang et al   | small CNN       | 60.1%    |      |
| [Scaling Provable Adversarial Defenses](https://arxiv.org/pdf/1805.12514.pdf) | Wong et al   | small CNN       | 56.90%   |      |
| (Verification) [A Convex Relaxation Barrier to Tight Robustness Verification of Neural Networks](https://arxiv.org/pdf/1902.08722.pdf) | Salman et al | CNN             | 39.83%   |      |



##### eps=0.4

| Defense/Verification                                         | Author           | Model Structure | Accuracy |      |
| ------------------------------------------------------------ | ---------------- | --------------- | -------- | ---- |
| [Towards Stable and Efficient Training of Verifiably Robust Neural Networks](https://arxiv.org/pdf/1906.06316.pdf) | Zhang et al   | large CNN | 87.04%   | pick the best number |
| [On the Effectiveness of Interval Bound Propagation for Training Verifiably Robust Models](https://arxiv.org/pdf/1810.12715.pdf) | Gowal et al      | CNN             | 85.12%   |      |
| [Fast and Stable Interval Bounds Propagation for Training Verifiably Robust Models](https://arxiv.org/pdf/1906.00628.pdf) | Morawiecki et al | large CNN       | 84.42%   |      |
| (Verification) [Evaluating Robustness of Neural Networks with Mixed Integer Programming](https://arxiv.org/abs/1711.07356) | Tjeng et al      | small CNN       | 51.02%   |      |





### SVHN

The image size is 32 x 32 x 3 (3-channel in color). Pixel colors in [0, 255]. When calculating eps, these values are rescaled to [0, 1].

#### L2

##### eps=0.1

| Defense/Verification                                         | Author        | Model Structure | Accuracy |                              |
| ------------------------------------------------------------ | ------------- | --------------- | -------- | ---------------------------- |
| [Certified Adversarial Robustness via Randomized Smoothing](https://arxiv.org/pdf/1902.02918.pdf) | Cohen et al   | Resnet-20       | ~95%     | Interpolate from Cohen et al |
| [Lipschitz-Margin Training: Scalable Certification of Perturbation Invariance for Deep Neural Networks](https://arxiv.org/pdf/1802.04034.pdf) | Tsuzuku et al | Resnet-20       | 0%       | Interpolate from Cohen et al |



##### eps=0.2

| Defense/Verification                                         | Author        | Model Structure | Accuracy |                              |
| ------------------------------------------------------------ | ------------- | --------------- | -------- | ---------------------------- |
| [Certified Adversarial Robustness via Randomized Smoothing](https://arxiv.org/pdf/1902.02918.pdf) | Cohen et al   | Resnet-20       | ~88%     | Interpolate from Cohen et al |
| [Lipschitz-Margin Training: Scalable Certification of Perturbation Invariance for Deep Neural Networks](https://arxiv.org/pdf/1802.04034.pdf) | Tsuzuku et al | Resnet-20       | 0%       | Interpolate from Cohen et al |



#### L-Infty

##### eps=0.01

| Defense/Verification                                         | Author          | Model Structure    | Accuracy |      |
| ------------------------------------------------------------ | --------------- | ------------------ | -------- | ---- |
| [Training Verified Learners with Learned Verifiers](https://arxiv.org/pdf/1805.10265.pdf) | Dvijotham et al | Predictor-Verifier | 62.44%   |      |
| [On the Effectiveness of Interval Bound Propagation for Training Verifiably Robust Models](https://arxiv.org/pdf/1810.12715.pdf) | Gowal et al     | CNN                | 62.40%   |      |
| [Provable Defenses against Adversarial Examples via the Convex Outer Adversarial Polytope](https://arxiv.org/pdf/1711.00851.pdf) | Wong et al      | CNN                | 59.33%   |      |
| [Differentiable Abstract Interpretation for Provably Robust Neural Networks](http://proceedings.mlr.press/v80/mirman18b/mirman18b.pdf) | Mirman et al    | small CNN          | 11.0%    |      |



##### eps=8/255

| Defense/Verification                                         | Author           | Model Structure | Accuracy |                            |
| ------------------------------------------------------------ | ---------------- | --------------- | -------- | -------------------------- |
| [On the Effectiveness of Interval Bound Propagation for Training Verifiably Robust Models](https://arxiv.org/pdf/1810.12715.pdf) | Gowal et al      | large CNN       | 47.63%   | Report by Morawiecki et al |
| [Fast and Stable Interval Bounds Propagation for Training Verifiably Robust Models](https://arxiv.org/pdf/1906.00628.pdf) | Morawiecki et al | small CNN       | 46.03%   |                            |



### Fashion-MNIST

This is a MNIST-like dataset. Images are 28 x 28 and grayscale. Values are in [0, 1].

#### L-Infty

##### eps=0.1

| Defense/Verification                                         | Author     | Model Structure | Accuracy |      |
| ------------------------------------------------------------ | ---------- | --------------- | -------- | ---- |
| [Towards Stable and Efficient Training of Verifiably Robust Neural Networks](https://arxiv.org/pdf/1906.06316.pdf) | Zhang et al   | large CNN | 78.73%   | pick the best number |
| [On the Effectiveness of Interval Bound Propagation for Training Verifiably Robust Models](https://arxiv.org/pdf/1810.12715.pdf) | Gowal et al      | large CNN             | 77.63%   | pick the best number, reported by Zhang et al |
| [Provable Defenses against Adversarial Examples via the Convex Outer Adversarial Polytope](https://arxiv.org/pdf/1711.00851.pdf) | Wong et al | CNN             | 65.47%   |      |



**\***. Within one dataset, L-2 and L-Infty balls are mutually transformable. After transformation, a corresponding tight bound may exist but not listed.

**Notes**:

1. Some papers use rarely-used epsilon to report their results, which increases comparison difficulty. Some papers use epsilon after regularization instead of raw one, which induces confusion.

   We think it is better to adapt common evaluation epsilons and routines.

2. Instead of evaluating on above benchmarks and reporting the robust accuracy, some papers tend to report average robust radius. When such papers become abundant, we will add comparison table for that metric.

3. Some verification works tend to use small toy models or random models for evaluation. The model it used limits the verified robustness it can achieve, no matter how tight it is. We suggest using trained robust models for evaluation.

4. Besides the on-the-board results, all these papers have their own unique takeaways. For interested reader and stackholders, we recommend not to only value the approach with highest number.

## Reference: Empirical Robustness

For comparison, here we cite numbers from MadryLab repositories for [MNIST challenge](https://github.com/MadryLab/mnist_challenge) and [CIFAR-10 challenge](https://github.com/MadryLab/cifar10_challenge), which records the best attacks towarding their robust model with secret weights.



### CIFAR-10

#### L-Infty

##### eps=8/255

*Block-Box*

| Attack                                                       | Submitted by    | Accuracy | Submission Date |
| ------------------------------------------------------------ | --------------- | -------- | --------------- |
| PGD on the cross-entropy loss for the adversarially trained public network | (initial entry) | 63.39%   | Jul 12, 2017    |

*White-Box*

| Attack        | Submitted by | Accuracy | Submission Date |
| ------------- | ------------ | -------- | --------------- |
| MultiTargeted | Sven Gowal   | 44.03%   | Aug 28, 2019    |



### MNIST

#### L-Infty

##### eps=0.3

*Black-Box*

| Attack                                                       | Submitted by | Accuracy | Submission Date |
| ------------------------------------------------------------ | ------------ | -------- | --------------- |
| AdvGAN from ["Generating Adversarial Examples with Adversarial Networks"](https://arxiv.org/abs/1801.02610) | AdvGAN       | 92.76%   | Sep 25, 2017    |

*White-Box*

| Attack                                         | Submitted by  | Accuracy | Submission Date |
| ---------------------------------------------- | ------------- | -------- | --------------- |
| First-Order Adversary with Quantized Gradients | Zhuanghua Liu | 88.32%   | Oct 16, 2019    |




## An (Incomplete) Paper List

Works in this field include provable training approaches and verification approaches. Related analysis or discussion papers are also listed.



**Exact Verifiers**

- [Reluplex: An Efficient SMT Solver for Verifying Deep Neural Networks](https://arxiv.org/abs/1702.01135)

  (CAV 2017, arxiv:1702.01135)

  Feb 2017

  Guy Katz, Clark Barrett, David Dill, Kyle Julian, Mykel Kochenderfer

- (Planet) [Formal Verification of Piece-Wise Linear Feed-Forward Neural Networks](https://arxiv.org/abs/1705.01320)

  (ISATVA 2017, arxiv: 1705.01320)

  May 2017

  Ruediger Ehlers

- (Survery paper) [Algorithms for Verifying Deep Neural Networks](https://arxiv.org/abs/1903.06758)

  (arxiv: 1903.06758)

  Mar 2019

  Changliu Liu, Tomer Arnon, Christopher Lazarus, Clark Barrett, Mykel J. Kochenderfer

**MILP** (Mixed Interger Programming)

A fast exact verifier.

- [Evaluating Robustness of Neural Networks with mixed Integer Programming](https://arxiv.org/abs/1711.07356)

  (ICLR 2019, arxiv:1711.07356)

  Nov 2017

  Vincent Tjeng, Kai Xiao, Russ Tedrake

- [Training for Faster Adversarial Robustness Verification via Inducing ReLU Stability](https://arxiv.org/abs/1809.03008)

  (ICLR 2019, arxiv: 1809.03008)

  Sept 2018

  Kai Y. Xiao, Vincent Tjeng, Nur Muhammad Shafiullah, Aleksander Madry

**Lipschitz-Based**

- (Heuristic, CLEVER) [Evaluating the Robustness of Neural Networks: An Extreme Value Theory Approach](https://arxiv.org/abs/1801.10578)

  (ICLR 2018, arxiv: 1801.10578)

  Jan 2018

  *Tsui-Wei Weng, *Huan Zhang, Pin-Yu Chen, Jinfeng Yi, Dong Su, Yupeng Gao, Cho-Jui Hsieh, Luca Daniel

- [Lipschitz-Margin Training: Scalable Certification of Perturbation Invariance for Deep Neural Networks](https://arxiv.org/abs/1802.04034)

  (NeurIPS 2018, arxiv: 1802.04034)

  Feb 2018

  Yusuke Tsuzuku, Issei Sato, Masashi Sugiyama

- (Fast-Lip) [Towards Fast Computation of Certified Robustness for ReLU Networks](https://arxiv.org/abs/1804.09699)

  (ICML 2018, arxiv: 1804.09699)

  Apr 2018

  *Tsui-Wei Weng, *Huan Zhang, Hongge Chen, Zhao Song, Cho-Jui Hsieh, Duane Boning, Inderjit S. Dhillon, Luca Daniel

- [On Extensions of CLEVER: A Neural Network Robustness Evaluation Algorithm](https://arxiv.org/abs/1810.08640)

  (GlobalSIP 2018, arxiv: 1810.08640)

  Oct 2018

  *Tsui-Wei Weng, *Huan Zhang, Pin-Yu Chen, Aurelie Lozano, Cho-Jui Hsieh, Luca Daniel

- [RecurJac: An Efficient Recursive Algorithm for Bounding Jacobian Matrix of Neural Networks and Its Applications](https://arxiv.org/abs/1810.11783)

  (AAAI 2019, arxiv: 1810.11783)

  Oct 2018

  Huan Zhang, Pengchuan Zhang, Cho-Jui Hsieh

**IBP** (Interval Bound Propagation)

- (IBP + Dual) [Training Verified Learners with Learned Verifiers](https://arxiv.org/abs/1805.10265)

  (arxiv: 1805.10265)

  May 2018

  Krishnamurthy Dvijotham, Sven Gowal, Robert Stanforth, Relja Arandjelovic, Brendan O'Donoghue, Jonathan Uesato, Pushmeet Kohli

- [On the Effectiveness of Interval Bound Propagation for Training Verifiably Robust Models](https://arxiv.org/abs/1810.12715)

  (NeurIPS 2018 Workshop Best Paper, arxiv: 1810.12715)

  Oct 2018

  Sven Gowal, Krishnamurthy Dvijotham, Robert Stanforth, Rudy Bunel, Chongli Qin, Jonathan Uesato, Relja Arandjelovic, Timothy Mann, Pushmeet Kohli

- [Fast and Stable Interval Bounds Propagation for Training Verifiably Robust Models](https://arxiv.org/abs/1906.00628)

  (arxiv: 1906.00628)

  Jun 2019

  Paweł Morawiecki, Przemysław Spurek, Marek Śmieja, Jacek Tabor
  
- [On the Effectiveness of Interval Bound Propagation for Training Verifiably Robust Models](https://arxiv.org/abs/1906.06316)

  (arxiv: 1906.06316)

  Jun 2019

  Huan Zhang, Hongge Chen, Chaowei Xiao, Bo Li, Duane Boning, Cho-Jui Hsieh
  

**Linear Relaxations**

- (Fast-Lin) [Towards Fast Computation of Certified Robustness for ReLU Networks](https://arxiv.org/abs/1804.09699)

  (ICML 2018, arxiv: 1804.09699)

  Apr 2018

  *Tsui-Wei Weng, *Huan Zhang, Hongge Chen, Zhao Song, Cho-Jui Hsieh, Duane Boning, Inderjit S. Dhillon, Luca Daniel

- (Zonotope) [Differentiable Abstract Interpretation for Provably Robust Neural Networks](http://proceedings.mlr.press/v80/mirman18b/mirman18b.pdf)

  (ICML 2018)

  Jul 2018

  Matthew Mirman, Timon Gehr, Martin Vechev

- (Zonotope) [Boosting Robustness Certification of Neural Networks](https://openreview.net/forum?id=HJgeEh09KQ)

  (ICLR 2019)

  Sep 2018

  Gagandeep Singh, Timon Gehr, Markus Püschel, Martin Vechev

- (CROWN) [Efficient Neural Network Robustness Certification with General Activation Functions](https://arxiv.org/abs/1811.00866)

  (NIPS 2018, arxiv: 1811.00866)

  Nov 2018

  *Huan Zhang, *Tsui-Wei Weng, Pin-Yu Chen, Cho-Jui Hsieh, Luca Daniel

- (Zonotope) [Fast and Effective Robustness Certification](https://papers.nips.cc/paper/8278-fast-and-effective-robustness-certification)

  (NIPS 2018)

  Dec 2018

  Gagandeep Singh, Timon Gehr, Matthew Mirman, Markus Püschel, Martin Vechev

- (Zonotope) [An Abstract Domain for Certifying Neural Networks](https://dl.acm.org/citation.cfm?id=3290354)

  (POPL 2019)

  Jan 2019

  Gagandeep Singh, Timon Gehr, Markus Püschel, Martin Vechev

- (Unification) [A Convex Relaxation Barrier to Tight Robust Verification of Neural Networks](https://arxiv.org/abs/1902.08722)

  (NeurIPS 2019, arxiv: 1902.08722)

  Feb 2019

  Hadi Salman, Greg Yang, Huan Zhang, Cho-Jui Hsieh, Pengchuan Zhang

- [On the Effectiveness of Interval Bound Propagation for Training Verifiably Robust Models](https://arxiv.org/abs/1906.06316)

  (arxiv: 1906.06316)

  Jun 2019

  Huan Zhang, Hongge Chen, Chaowei Xiao, Bo Li, Duane Boning, Cho-Jui Hsieh

**Linear Dual Space Relaxations**

- [Provable Defenses against Adversarial Examples via the Convex Outer Adversarial Polytope](https://arxiv.org/abs/1711.00851)

  (ICML 2018, arxiv: 1711.00851)

  Nov 2017

  Eric Wong, J. Zico Kolter

- [A Dual Approach to Scalable Verification of Deep Networks](https://arxiv.org/abs/1803.06567)

  (UAI 2018 Best Paper, arxiv: 1803.06567)

  Mar 2018

  Krishnamurthy (Dj)Dvijotham, Robert Stanforth, Sven Gowal, Timothy Mann, Pushmeet Kohli

- [Scaling Provable Adversarial Defenses](https://arxiv.org/abs/1805.12514)

  (NIPS 2018, arxiv: 1805.12514)

  May 2018

  Eric Wong, Frank R. Schmidt, Jan Hendrik Metzen, J. Zico Kolter

- [Robustra: Training Provable Robust Neural Networks over Reference Adversarial Space](https://www.ijcai.org/proceedings/2019/654)

  (IJCAI 2019)

  *Linyi Li, *Zexuan Zhong, Bo Li, Tao Xie

**SDP and SDP-Dual**

- [Certified Defenses against Adversarial Examples](https://arxiv.org/abs/1801.09344)

  (ICLR 2018, arxiv: 1801.09344)

  Jan 2018

  Aditi Raghunathan, Jacob Steinhardt, Percy Liang

- [Semidefinite relaxations for certifying robustness to adversarial examples](https://arxiv.org/abs/1811.01057)

  (NIPS 2018, arxiv: 1811.01057)

  Nov 2018

  Aditi Raghunathan, Jacob Steinhardt, Percy Liang

- [Safety Verification and Robustness Analysis of Neural Networks via Quadratic Constraints and Semidefinite Programming](https://arxiv.org/abs/1903.01287)

  (arxiv: 1903.01287)

  Mar 2019

  Mahyar Fazlyab, Manfred Morari, George J. Pappas

- (SDP for Lipschitz) [Efficient and Accurate Estimation of Lipschitz Constants for Deep Neural Networks](https://arxiv.org/abs/1906.04893)

  (NeurIPS 2019, arxiv: 1906.04893)

  Jun 2019

  Mahyar Fazlyab, Alexander Robey, Hamed Hassani, Manfred Morari, George J. Pappas

- [Efficient Neural Network Verification with Exactness Characterization](http://auai.org/uai2019/proceedings/papers/164.pdf)

  (UAI 2019, paper 164)

  Jul 2019

  Krishnamurthy (Dj) Dvijotham, Robert Stanforth, Sven Gowal, Chongli Qin, Soham De, Pushmeet Kohli

**Differential Privacy and Randomized Smoothing**

- [Certified Robustness to Adversarial Examples with Differential Privacy](https://arxiv.org/abs/1802.03471)

  (S&P 2019, arxiv: 1802.03471)

  Feb 2018

  Mathias Lecuyer, Vaggelis Atlidakis, Roxana Geambasu, Daniel Hsu, Suman Jana

- [Certified Adversarial Robustness via Randomized Smoothing](https://arxiv.org/abs/1902.02918)

  (ICML 2019, arxiv: 1902.02918)

  Feb 2019

  Jeremy M Cohen, Elan Rosenfeld, J. Zico Kolter

- [Provably Robust Deep Learning via Adversarially Trained Smoothed Classifiers](https://arxiv.org/abs/1906.04584)

  (NeurIPS 2019, arxiv: 1906.04584)

  Jun 2019

  Hadi Salman, Greg Yang, Jerry Li, Pengchuan Zhang, Huan Zhang, Ilya Razenshteyn, Sebastien Bubeck

- [A Stratified Approach to Robustness for Randomly Smoothed Classifiers](https://arxiv.org/abs/1906.04948)

  (arxiv: 1906.04948)

  Jun 2019

  Guang-He Lee, Yang Yuan, Shiyu Chang, Tommi S. Jaakkola

**Hybrid**

- (Reluval) [Formal security analysis of neural networks using symbolic intervals](https://arxiv.org/abs/1804.10829)

  (USENIX security 2018, arxiv: 1804.10829)

  Apr 2018

  Shiqi Wang, Kexin Pei, Justin Whitehouse, Junfeng Yang, Suman Jana

- (Zonotope) [AI2: Safety and Robustness Certification of Neural Networks with Abstract Interpretation](https://ieeexplore.ieee.org/document/8418593)

  (S&P 2018)

  May 2018

  Timon Gehr, Matthew Mirman, Dana Drachsler-Cohen, Petar Tsankov, Swarat Chaudhuri, Martin Vechev

- [Optimization + Abstraction: A Synergistic Approach for Analyzing Neural Network Robustness](https://arxiv.org/abs/1904.09959)

  (PLDI 2019, arxiv: 1904.09959)

  Apr 2019

  Greg Anderson, Shankara Pailoor, Isil Dillig, Swarat Chaudhuri

**Ensemble**


- (Cascade)[Scaling Provable Adversarial Defenses](https://arxiv.org/abs/1805.12514)

  (NIPS 2018, arxiv: 1805.12514)

  May 2018

  Eric Wong, Frank R. Schmidt, Jan Hendrik Metzen, J. Zico Kolter

- (Cascade)[Enhancing Certifiable Robustness via a Deep Model Ensemble](https://arxiv.org/abs/1910.14655)

  (ICLR 2019 Workshop, arxiv: 1910.14655)

  Oct 2019

  Huan Zhang, Minhao Cheng, Cho-Jui Hsieh

**Distributional and Probabilistic**

- [Certifying some distributional robustness with principled adversarial training](https://arxiv.org/abs/1710.10571)

  (ICLR 2018, arxiv: 1710.10571)

  Oct 2017

  Aman Sinha, Hongseok Namkoong, John Duchi

- [PROVEN: Certifying Robustness of Neural Networks with a Probabilistic Approach](https://arxiv.org/abs/1812.08329)

  (ICML 2019, arxiv: 1812.08329)

  Dec 2018

  Tsui-Wei Weng, Pin-Yu Chen, Lam M. Nguyen, Mark S. Squillante, Ivan Oseledets, Luca Daniel

**Theory and Analysis**

- (Lp Bound Unreliable) [On the sensitivity of adversarial robustness to input data distributions](https://arxiv.org/abs/1902.08336)

  (ICLR 2019, arxiv: 1902.08336)

  Feb 2019

  Gavin Weiguang Ding, Kry Yik Chau Lui, Xiaomeng Jin, Luyu Wang, Ruitong Huang

- [Universal Approximation with Certified Networks](https://arxiv.org/abs/1909.13846)

  (ICLR 2020 Submission, arxiv:1909.13846)

  Sep 2019

  Maximilian Baader, Matthew Mirman, Martin Vechev

**Other Approches**

- [Provable Robustness of ReLU networks via Maximization of Linear Regions](https://arxiv.org/abs/1810.07481)

  (AISTATS 2019, arxiv: 1810.07481)

  Oct 2018

  Francesco Croce, Maksym Andriushchenko, Matthias Hein

- [Provable Certificates for Adversarial Examples: Fitting a Ball in the Union of Polytopes](https://arxiv.org/abs/1903.08778)

  (ICML 2019 SPML Workshop, ArXiv: 1903.08778)

  Mar 2019

  Matt Jordan, Justin Lewis, Alexandros G. Dimakis

**Dealing with General Settings**

Note that many papers above can be generalized to activation functions beyond ReLU. Here we only list those which deem their main contribution as dealing with general settings.

- (Zonotope) [An Abstract Domain for Certifying Neural Networks](https://dl.acm.org/citation.cfm?id=3290354)

  (POPL 2019)

  Jan 2019

  Gagandeep Singh, Timon Gehr, Markus Püschel, Martin Vechev

- (Non-linear Specs) [Verification of Non-Linear Specifications for Neural Networks](https://arxiv.org/abs/1902.09592)

  (ICLR 2019, arxiv: 1902.09592)

  Feb 2019

  Chongli Qin, Krishnamurthy (Dj)Dvijotham, Brendan O'Donoghue, Rudy Bunel, Robert Stanforth, Sven Gowal, Jonathan Uesato, Grzegorz Swirszcz, Pushmeet Kohli

---

### Applications

The following papers either apply the above approaches to specific domains, or deal with different but closely related problems.

**NLP**

- [Certified Robustness to Adversarial Word Substitutions](https://arxiv.org/abs/1909.00986)

  (EMNLP 2019, arxiv: 1909.00986)

  Sep 2019

  Robin Jia, Aditi Raghunathan, Kerem Göksel, Percy Liang

- [Achieving Verified Robustness to Symbol Substitutions via Interval Bound Propagation](https://arxiv.org/abs/1909.01492)

  (arxiv: 1909.01492)

  Sep 2019

  Po-Sen Huang, Robert Stanforth, Johannes Welbl, Chris Dyer, Dani Yogatama, Sven Gowal, Krishnamurthy Dvijotham, Pushmeet Kohli

**Tree Model**

- [Robustness Verification of Tree-based Models](https://arxiv.org/abs/1906.03849)

  (NeurIPS 2019, arxiv: 1906.03849)

  Jun 2019

  Hongge Chen, Huan Zhang, Si Si, Yang Li, Duane Boning, Cho-Jui Hsieh

**CNN**

- [CNN-Cert: An Efficient Framework for Certifying Robustness of Convolutional Neural Networks](https://arxiv.org/abs/1811.12395)

  (arxiv: 1811.12395)

  Nov 2018

  Akhilan Boopathy, Tsui-Wei Weng, Pin-Yu Chen, Sijia Liu, Luca Daniel

**CV**

- (blackbox CV) [Towards Practical Verification of Machine Learning: The Case of Computer Vision Systems](https://arxiv.org/abs/1712.01785)

  (arxiv: 1712.01785)

  Dec 2017

  Kexin Pei, Yinzhi Cao, Junfeng Yang, Suman Jana

**Reinforcement Learning**

- (Policy Verify) [Verification of Neural Network Control Policy Under Persistent Adversarial Perturbation](https://arxiv.org/abs/1908.06353)

  (arxiv:1908.06353)

  Aug 2019

  Yuh-Shyang Wang, Tsui-Wei Weng, Luca Daniel

**Probabilistic Models**

- [Verification of Deep Probabilistic models](https://arxiv.org/abs/1812.02795)

  (NIPS 2018 Workshop, arxiv: 1812.02795)

  Dec 2018

  Krishnamurthy Dvijotham, Marta Garnelo, Alhussein Fawzi, Pushmeet Kohli

---

Maintained by Linyi.

Last updated: Oct 28, 2019
