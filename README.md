# Awesome Provable Robust Neural Networks
Recently, provable adversarial robustness and robustness verfication becomes a hot research topic.

In constrast to empirical robustness and empirical adversarial attacks, (common) provable robustness neural networks give a rigorous guarantee about the successful rate bound, that **no** existing or future attacks will break. Robustness verification approaches stongly connect with provable robustness, which verifies such successful rate bound given a neural network model.

Trained on training set, the provable robustness is often measure by **robust accuracy** or **error rate** on the test set. One sample is considered accurate if and only if we can prove that there is no adversararial samples exist in the neighborhood, i.e., the model always outputs the current prediction label in the neighborhood. The neighborhood is usually defined by L-norm balls. For example, L-infty balls are $\{x + \delta: \lVert \delta \rVert_\infty \le \epsilon\}$ , L-2 balls are $\{x + \delta: \lVert \delta \rVert_2 \le \epsilon\}$. The size of the ball is controlled by eps ($\epsilon$).

Better provable robustness can be achieved by better provable robust training approaches which yield more robust models, tighter robustness verification approaches which yield tighter guarantees, or jointly.

#### Scoop of the Repo

Currently, works in literature mainly focuses on image classification tasks with datasets MNIST, CIFAR10, ImageNet, FashionMNIST, and SVHN.

Mainly perturbation norm are L-2 balls and l-infty balls.

This repo mainly records recent progress of above settings, while advances in other settings are recorded in the attached paperlist.

We only consider single model robustness.

#### Contact & Updates

We are trying to keep track of all important advances in provable robustness, but may still miss some works.

Please feel free to contact us (Linyi(linyi2@illinois.edu) @ [UIUC Secure Learning Lab](https://aisecure.github.io/)) or commit your updates :)

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
| (Verification) [An Abstract Domain for Certifying Neural Networks](https://files.sri.inf.ethz.ch/website/papers/DeepPoly.pdf) | Singh et al     | CNN             | 40%         | ~2.04/255, only evaluated from 100 samples among all 10,000  |
| [Training for Faster Adversarial Robustness Verification via Inducing ReLU Stability](https://arxiv.org/pdf/1809.03008.pdf) | Xiao et al      | CNN             | 45.93%      |                                                              |
| (Verification)[A Dual Approach to Scalable Verification of Deep Networks](http://auai.org/uai2018/proceedings/papers/204.pdf) | Dvijotham et al | CNN             | 20%         | LP-Dual verification on Uesato et al. model; interpolated from Fig. 2(a) |



##### eps=8/255

| Defense/Verification                                         | Author           | Model Structure    | Accuracy |                                                              |
| ------------------------------------------------------------ | ---------------- | ------------------ | -------- | ------------------------------------------------------------ |
| [Fast and Stable Interval Bounds Propagation for Training Verifiably Robust Models](https://arxiv.org/pdf/1906.00628.pdf) | Morawiecki et al | Small CNN          | 39.88%   |                                                              |
| [Differentiable Abstract Interpretation for Provably Robust Neural Networks](http://proceedings.mlr.press/v80/mirman18b/mirman18b.pdf) | Mirman et al     | Small CNN          | 37.4%    | ~7.65/255, only evaluated from 500 samples among all 10,000  |
| [On the Effectiveness of Interval Bound Propagation for Training Verifiably Robust Models](https://arxiv.org/pdf/1810.12715.pdf) | Gowal et al      | CNN                | 32.04%   |                                                              |
| [Training Verified Learners with Learned Verifiers](https://arxiv.org/pdf/1805.10265.pdf) | Dvijotham et al  | Predictor-Verifier | 26.67%   |                                                              |
| [Robustra: Training Provable Robust Neural Networks over Reference Adversarial Space](https://www.ijcai.org/proceedings/2019/654) | Li et al         | CNN                | 25.13%   |                                                              |
| [A Provable Defense for Deep Residual Networks](https://arxiv.org/pdf/1903.12519.pdf) | Mirman et al     | ResNet-Tiny        | 23.2%    |                                                              |
| [Evaluating Robustness of Neural Networks with Mixed Integer Programming](https://arxiv.org/pdf/1711.07356.pdf) | Tjeng et al      | CNN                | 22.40%   |                                                              |
| [Scaling provable adversarial defenses](https://arxiv.org/pdf/1902.02918.pdf) | Wong et al       | Resnet             | 21.78%   |                                                              |
| (Verification) [Boosting Robustness Certification of Neural Networks](https://files.sri.inf.ethz.ch/website/papers/DeepZ.pdf) | Singh et al      | Small CNN          | 21%      | ~7.65/255, only evaluated from 500 samples among all 10,000  |
| [Training for Faster Adversarial Robustness Verification via Inducing ReLU Stability](https://arxiv.org/pdf/1809.03008.pdf) | Xiao et al       | CNN                | 20.27%   |                                                              |
| (Verification) [A Dual Approach to Scalable Verification of Deep Networks](http://auai.org/uai2018/proceedings/papers/204.pdf) | Dvijotham et al  | CNN                | 0%       | LP-Dual verification on Uesato et al. model; interpolated from Fig. 2(a) |



### MNIST

All input images are gray-scale; 28 x 28 size; each pixel is in range [0, 1].

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
| [Training Verified Learners with Learned Verifiers](https://arxiv.org/pdf/1805.10265.pdf) | Dvijotham et al   | Predictor-Verifier | 95.56%   |                                                  |
| [Provable Defenses against Adversarial Examples via the Convex Outer Adversarial Polytope](https://arxiv.org/pdf/1711.00851.pdf) | Wong et al        | CNN                | 94.18%   |                                                  |
| (Verification) [Efficient Neural Network Verification with Exactness Characterization](http://auai.org/uai2019/proceedings/papers/164.pdf) | Dvijotham et al   | Grad-NN            | 83.68%   |                                                  |
| (Verification) [A Convex Relaxation Barrier to Tight Robustness Verification of Neural Networks](https://arxiv.org/pdf/1902.08722.pdf) | Salman et al      | CNN                | 82.76%   |                                                  |
| (Verification) [Semidefinite relaxations for certifying robustness to adversarial examples](https://arxiv.org/pdf/1811.01057.pdf) | Raghunathan et al | small NN           | 82%      |                                                  |
| [Certified Defenses against Adversarial Examples](https://arxiv.org/pdf/1801.09344.pdf) | Raghunathan et al | 2-layer NN         | 65%      |                                                  |



##### eps=0.3

| Defense/Verification                                         | Author       | Model Structure | Accuracy |      |
| ------------------------------------------------------------ | ------------ | --------------- | -------- | ---- |
| [On the Effectiveness of Interval Bound Propagation for Training Verifiably Robust Models](https://arxiv.org/pdf/1810.12715.pdf) | Gowal et al  | CNN             | 91.95%   |      |
| [Robustra: Training Provable Robust Neural Networks over Reference Adversarial Space](https://www.ijcai.org/proceedings/2019/0654.pdf) | Li et al     | small CNN       | 83.09%   |      |
| (Verification) [Evaluating Robustness of Neural Networks with Mixed Integer Programming](https://arxiv.org/abs/1711.07356) | Tjeng et al  | large CNN       | 75.81%   |      |
| [MixTrain: Scalable Training of Verifiably Robust Neural Networks](https://arxiv.org/pdf/1811.02625.pdf) | Wang et al   | small CNN       | 60.1%    |      |
| [Scaling Provable Adversarial Defenses](https://arxiv.org/pdf/1805.12514.pdf) | Wong et al   | small CNN       | 56.90%   |      |
| (Verification) [A Convex Relaxation Barrier to Tight Robustness Verification of Neural Networks](https://arxiv.org/pdf/1902.08722.pdf) | Salman et al | CNN             | 39.83%   |      |



##### eps=0.4

| Defense/Verification                                         | Author           | Model Structure | Accuracy |      |
| ------------------------------------------------------------ | ---------------- | --------------- | -------- | ---- |
| [On the Effectiveness of Interval Bound Propagation for Training Verifiably Robust Models](https://arxiv.org/pdf/1810.12715.pdf) | Gowal et al      | CNN             | 85.12%   |      |
| [Fast and Stable Interval Bounds Propagation for Training Verifiably Robust Models](https://arxiv.org/pdf/1906.00628.pdf) | Morawiecki et al | large CNN       | 84.42%   |      |
| (Verification) [Evaluating Robustness of Neural Networks with Mixed Integer Programming](https://arxiv.org/abs/1711.07356) | Tjeng et al      | small CNN       | 51.02%   |      |





### SVHN

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

#### L-Infty

##### eps=0.1

| Defense/Verification                                         | Author     | Model Structure | Accuracy |      |
| ------------------------------------------------------------ | ---------- | --------------- | -------- | ---- |
| [Provable Defenses against Adversarial Examples via the Convex Outer Adversarial Polytope](https://arxiv.org/pdf/1711.00851.pdf) | Wong et al | CNN             | 65.47%   |      |



**\***. Within one dataset, L-2 and L-Infty balls are mutually transformable. After transformation, a corresponding tight bound may exist but not listed.

**Notes**:

1. Some papers use rarely-used epsilon to report their results, which increases comparison difficulty. Some papers use epsilon after regularization instead of raw one, which induces confusion.

   We think it is better to adapt common evaluation epsilons and routines.

2. Instead of evaluating on above benchmarks and reporting the robust accuracy, some papers tend to report average robust radius. When such papers become abundant, we will add comparison table for that metric.

3. Some verification works tend to use small toy models or random models for evaluation. The model it used limits the verified robustness it can achieve, no matter how tight it is. We suggest using trained robust models for evaluation.

4. Besides the on-the-board results, all these papers have their own unique takeaways. For interested reader and stackholders, we recommend not to only value the approach with highest number.

## Reference: Empirical Robustness Bound


## An (Incomplete) Paper List

Works in this field include provable training approaches and verification approaches. Related analysis or discussion papers are also listed.



Exact Verifiers:

- Reluplex: An Efficient SMT Solver for Verifying Deep Neural Networks(CAV 2017, arxiv:1702.01135)
- (Planet) Formal Verification of Piece-Wise Linear Feed-Forward Neural Networks(ISATVA 2017, arxiv: 1705.01320)First proposes ReLU linear relaxations. While don’t know if used.
- Algorithms for Verifying Deep Neural Networks(arxiv: 1903.06758)This survey covers many earlier-stage SMT-based works, but few latest works in 2018 and 2019.

MILP:

A fast exact verifier.

- Evaluating Robustness of Neural Networks with mixed Integer Programming(ICLR 2019, arxiv:1711.07356)State-of-the-art of exact verifier
- Training for Faster Adversarial Robustness Verification via Inducing ReLU Stability(ICLR 2019, arxiv: 1809.03008)On training side, regularization based. 19.32% on MNIST eps=0.3 l_infty ball.

Lipschitz-Based:

- (Heuristic, CLEVER) Evaluating the Robustness of Neural Networks: An Extreme Value Theory Approach(ICLR 2018, arxiv: 1801.10578)
- On Extensions of CLEVER: A Neural Network Robustness Evaluation Algorithm(arxiv: 1810.08640)A short extension
- (Fast-Lip in)Towards Fast Computation of Certified Robustness for ReLU Networks(ICML 2018, arxiv: 1804.09699)Relatively loose
- RecurJac: An Efficient Recursive Algorithm for Bounding Jacobian Matrix of Neural Networks and Its Applications(AAAI 2019, arxiv: 1810.11783)
- Lipschitz-Margin Training: Scalable Certification of Perturbation Invariance for Deep Neural Networks(NeurIPS 2018, arxiv: 1802.04034)

IBP:

- On the Effectiveness of Interval Bound Propagation for Training Verifiably Robust Models(NeurIPS 2018 Workshop Best Paper, arxiv: 1810.12715)
- Fast and Stable Interval Bounds Propagation for Training Verifiably Robust Models(arxiv: 1906.00628)

Linear Relaxations:

- (Fast-Lin in)Towards Fast Computation of Certified Robustness for ReLU Networks(ICML 2018, arxiv: 1804.09699)By linear inequality propagation
- CROWN: Efficient Neural Network Robustness Certification with General Activation Functions(NIPS 2018 - 7742)Generalize the linear inequality propagation
- (Zonotope) Differentiable Abstract Interpretation for Provably Robust Neural Networks(ICML 2018)Substantially, zonotope is a compressed and efficient expression of fixed-formed linear equalities.
- (Zonotope) Fast and Effective Robustness Certification(NIPS 2018)Shrink the zonotope region
- (Zonotope) Boosting Robustness Certification of Neural Networks(ICLR 2019)Combine with MILP and LP to shrink interval bound on each neuron
- (Zonotope) An Abstract Domain for Certifying Neural Networks(POPL 2019)Generalize to support rotations.
- (Unification) A Convex Relaxation Barrier to Tight Robust Verification of Neural Networks(ArXiv: 1902.08722)A unification view of all LP and LP-Dual techniques.

Linear Dual Space Relaxations:

- Provable Defenses against Adversarial Examples via the Convex Outer Adversarial Polytope(ICML 2018, ArXiv: 1711.00851)First LP-Dual approach.
- Scaling Provable Adversarial Defenses(NIPS 2018, ArXiv: 1805.12514)Scaling the previous approach by probabilistic training.
- Efficient Neural Network Verification with Exactness Characterization(UAI 2018 Best Paper)Solve the optimal LP-bound. Be summarized in ArXiv: 1902.08722. But the concrete techniques to be read.
- Robustra: Training Provable Robust Neural Networks over Reference Adversarial Space(IJCAI 2019)LP-Dual training on the reference adversarial space.

SDP and SDP-Dual:

- Certified Defenses against Adversarial Examples(ICLR 2018)Strictly, it is not SDP. But it induces the following true SDP one and shares the same core idea.Only for two-layer networks.
- Semidefinite relaxations for certifying robustness to adversarial examples(NIPS 2018, Arxiv: 1811.01057)SDP based on relax xx^T to an arbitrary symmetric matrix allowing full-rank.
- Safety Verification and Robustness Analysis of Neural Networks via Quadratic Constraints and Semidefinite Programming(Arxiv: 1903.01287)An alternative way for SDP. Experiment results questionable.
- Efficient and Accurate Estimation of Lipschitz Constants for Deep Neural Networks(Arxiv: 1906.04893)Only for Lipschitz, but in fact the same method as the previous one.
- Efficient Neural Network Verification with Exactness Characterization(UAI 2019, paper 164)A SDP-dual approach which is accurate and fast.

Hybrid:

- (Zonotope) AI2: Safety and Robustness Certification of Neural Networks with Abstract Interpretation(S&P 2018)
- Formal security analysis of neural networks using symbolic intervals(USENIX security 2018)

Distributional and Probabilistic:

- Certifying some distributional robustness with principled adversarial training(ICLR 2018, ArXiv: 1710.10571)Another perspective and framework about certified robustness.
- PROVEN: Certifying Robustness of Neural Networks with a Probabilistic Approach(ICML 2019, ArXiv: 1812.08329)Tight probabilistic bound for existing verification techniques.

Differential Privacy and Randomized Smoothing:

- Certified Robustness to Adversarial Examples with Differential Privacy(S&P 2019, ArXiv: 1802.03471)Pioneering work, though beaten by Zico’s. Smoothing and provide bounds from differential privacy perspective.
- Certified Adversarial Robustness via Randomized Smoothing(ICML 2019, ArXiv: 1902.02918)Initial work on randomized smoothing.
- Provably Robust Deep Learning via Adversarially Trained Smoothed Classifiers(NeurIPS 2019, ArXiv: 1906.04584)Adversarial training of randomized smoothing. Also provide an alternate proof. State-of-the-art.
- A Stratified Approach to Robustness for Randomly Smoothed Classifiers(ArXiv: 1906.04948)Extend to discrete case with l-0 norm

Theory and Analysis:

-  (Lp Bound Unreliable) On the sensitivity of adversarial robustness to input data distributions(ICLR 2019, ArXiv: 1902.08336)

- Universal Approximation with Certified Networks(ICLR 2020 Submission, arXiv:1909.13846)

Applications:

NLP:

- Certified Robustness to Adversarial Word Substitutions(EMNLP 2019, arxiv: 1909.00986)

- Achieving Verified Robustness to Symbol Substitutions via Interval Bound Propagation(ArXiv: 1909.01492)

Tree Model:

- Robustness Verification of Tree-based Models		(NeurIPS 2019, ArXiv: 1906.03849)		Tight verification of tree-based models.

CNN:

-  CNN-Cert: An Efficient Framework for Certifying Robustness of Convolutional Neural Networks(ArXiv: 1811.12395)		Verify of CNN.

CV:

- (blackbox CV) Towards Practical Verification of Machine Learning: The Case of Computer Vision Systems(ArXiv: 1712.01785)

Reinforcement Learning:

- (Policy Verify) Verification of Neural Network Control Policy Under Persistent Adversarial Perturbation(arXiv:1908.06353)

Probabilistic Models:

- Verification of Deep Probabilistic models(NeurIPS 2018 Workshop, arXiv: 1812.02795)



Specification beyond L-balls:

- (Zonotope) An Abstract Domain for Certifying Neural Networks (POPL 2019)

  Generalize to support rotations.

- (Non-linear Specs) Verification of Non-Linear Specifications for Neural Networks(ICLR 2019, ArXiv: 1902.09592)

  A general approach to deal with non-linear specifications, including semantic distances.



Other Approches:

- Provable Robustness of ReLU networks via Maximization of Linear Regions

  (AISTATS 2019, ArXiv: 1810.07481)

- Provable Certificates for Adversarial Examples: Fitting a Ball in the Union of Polytopes

  (ICML 2019 SPML Workshop, ArXiv: 1903.08778)



Maintained by Linyi.

Last updated: Oct 28, 2019
