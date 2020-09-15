# Provable Training and Verification Approaches Towards Robust Neural Networks
Recently, provable (i.e. certified) adversarial robustness training and verification methods have demonstrated their effectiveness against adversarial attacks. In contrast to empirical robustness and empirical adversarial attacks, the provable robustness verification provides rigorous lower bound of robustness for a given neural network, such that **no** existing or future attacks will attack further. 

Note that the training methods towards robust networks are usually connected with the corresponding verification approach. For instance, after training, the robustness bound is often measure on the test set in terms of "robust accuracy"(RACC). One data sample is considered to be provable robust if and only if we can prove that there is no adversarial samples exist in the neighborhood, i.e., the model always outputs the current prediction label in the neighborhood. The neighborhood is usually defined by L-norm distance. 

Tighter provable robustness bound can be achieved by better robust training approaches, and tighter robustness verification approaches, or jointly.



**News:**

- We are happy to announce the FIRST large-scale study of representative certifiably robust defenses with interesting insights @ https://arxiv.org/abs/2009.04131! (It is also a useful paper list for certified robustness of DNNs)
- We also release a unified toolbox **VeriGauge** for implementing robustness verification approaches conveniently with PyTorch: https://github.com/AI-secure/VeriGauge
  Feel free to try it and give us feedback!
- [**new**] We include a taxnomy tree of representative approaches in this field (adapted from our large-scale study paper) at the bottom.



**Table of Contents**

------

- [Scope of the Repo](#scope-of-the-repo)


* [Contact & Updates](#contact---updates)
- [Main Leaderboard](#main-leaderboard)
  + [ImageNet](#imagenet)
    - [L2](#l2)
      * [eps = 0.2](#eps---02)
      * [eps = 0.5](#eps---05)
      * [eps = 1.0](#eps---10)
      * [eps = 2.0](#eps---20)
      * [eps = 3.0](#eps---30)
    - [L-Infty](#l-infty)
      * [eps=1/255](#eps-1-255)
      * [eps=1.785/255](#eps-1785-255)
  + [CIFAR-10](#cifar-10)
    - [L2](#l2-1)
      * [eps=0.14](#eps-014)
      * [eps=0.25](#eps-025)
      * [eps=0.5](#eps-05)
      * [eps=1.0](#eps-10)
      * [eps=1.5](#eps-15)
    - [L Infty](#l-infty)
      * [eps=2/255](#eps-2-255)
      * [eps=8/255](#eps-8-255)
  + [MNIST](#mnist)
    - [L2](#l2-2)
      * [eps=1.58](#eps-158)
    - [L-Infty](#l-infty-1)
      * [eps=0.1](#eps-01)
      * [eps=0.3](#eps-03)
      * [eps=0.4](#eps-04)
  + [SVHN](#svhn)
    - [L2](#l2-3)
      * [eps=0.1](#eps-01-1)
      * [eps=0.2](#eps-02)
    - [L-Infty](#l-infty-2)
      * [eps=0.01](#eps-001)
      * [eps=8/255](#eps-8-255-1)
  + [Fashion-MNIST](#fashion-mnist)
    - [L-Infty](#l-infty-3)
      * [eps=0.1](#eps-01-2)
- [Reference: Empirical Robustness](#reference--empirical-robustness)
  + [CIFAR-10](#cifar-10-1)
    - [L-Infty](#l-infty-4)
      * [eps=8/255](#eps-8-255-2)
  + [MNIST](#mnist-1)
    - [L-Infty](#l-infty-5)
      * [eps=0.3](#eps-03-1)
- [Taxnomy Tree](#taxnomy-tree)



## [Scope of the Repo](scope-of-the-repo)

Current works mainly focus on image classification tasks with datasets MNIST, CIFAR10, ImageNet, FashionMNIST, and SVHN.

We focus on perturbation measured by L-2 and L-infty norms.

This repo mainly records recent progress of above settings, while advances in other settings are recorded in the attached paperlist.

We only consider **single model** robustness.



## Contact & Updates

We are trying to keep track of all important advances of provable robustness approaches, but may still miss some.

Please feel free to contact us (Linyi(linyi2@illinois.edu) @ [UIUC Secure Learning Lab](https://aisecure.github.io/) & [Illinois ASE Group](http://asenews.blogspot.com/)) or commit your updates :)



## Main Leaderboard

### ImageNet

All input images contain three channels; each pixel is in range [0, 255].

#### L2

##### eps = 0.2

| Defense                                                      | Author        | Model Structure | RACC |      |
| ------------------------------------------------------------ | ------------- | --------------- | ---- | ---- |
| [Certified Robustness to Adversarial Examples with Differential Privacy](https://arxiv.org/pdf/1802.03471.pdf) | Lecuyer et al | Inception V3    | 40%  |      |



##### eps = 0.5

| Defense                                                      | Author       | Model Structure | RACC |      |
| ------------------------------------------------------------ | ------------ | --------------- | ---- | ---- |
| [MACER: Attack-free and Scalable Robust Training via Maximizing Certified Radius](https://arxiv.org/pdf/2001.02378.pdf) | Zhai et al   | ResNet-50       | 57%  |      |
| [Provably Robust Deep Learning via Adversarially Trained Smoothed Classifiers](https://arxiv.org/pdf/1906.04584.pdf) | Salman et al | ResNet-50       | 56%  |      |
| [Certified Adversarial Robustness via Randomized Smoothing](https://arxiv.org/pdf/1902.02918.pdf) | Cohen et al  | ResNet-50       | 49%  |      |
| [Consistency Regularization for Certified Robustness of Smoothed Classifiers](https://arxiv.org/pdf/2006.04062.pdf) | Jeong et al  | ResNet-50       | 48%  |      |

All above approaches use Randomized Smoothing (Cohen et al) to derive certification, with wrong probability at 0.1%.



##### eps = 1.0

| Defense                                                      | Author       | Model Structure | RACC |      |
| ------------------------------------------------------------ | ------------ | --------------- | ---- | ---- |
| [MACER: Attack-free and Scalable Robust Training via Maximizing Certified Radius](https://arxiv.org/pdf/2001.02378.pdf) | Zhai et al   | ResNet-50       | 43%  |      |
| [Provably Robust Deep Learning via Adversarially Trained Smoothed Classifiers](https://arxiv.org/pdf/1906.04584.pdf) | Salman et al | ResNet-50       | 43%  |      |
| [Consistency Regularization for Certified Robustness of Smoothed Classifiers](https://arxiv.org/pdf/2006.04062.pdf) | Jeong et al  | ResNet-50       | 41%  |      |
| [Certified Adversarial Robustness via Randomized Smoothing](https://arxiv.org/pdf/1902.02918.pdf) | Cohen et al  | ResNet-50       | 37%  |      |

All above approaches use Randomized Smoothing (Cohen et al) to derive certification, with wrong probability at 0.1%.



##### eps = 2.0

| Defense                                                      | Author       | Model Structure | RACC |      |
| ------------------------------------------------------------ | ------------ | --------------- | ---- | ---- |
| [Provably Robust Deep Learning via Adversarially Trained Smoothed Classifiers](https://arxiv.org/pdf/1906.04584.pdf) | Salman et al | ResNet-50       | 27%  |      |
| [Consistency Regularization for Certified Robustness of Smoothed Classifiers](https://arxiv.org/pdf/2006.04062.pdf) | Jeong et al  | ResNet-50       | 25%  |      |
| [MACER: Attack-free and Scalable Robust Training via Maximizing Certified Radius](https://arxiv.org/pdf/2001.02378.pdf) | Zhai et al   | ResNet-50       | 25%  |      |
| [Certified Adversarial Robustness via Randomized Smoothing](https://arxiv.org/pdf/1902.02918.pdf) | Cohen et al  | ResNet-50       | 19%  |      |

All above approaches use Randomized Smoothing (Cohen et al) to derive certification, with wrong probability at 0.1%.



##### eps = 3.0

| Defense                                                      | Author       | Model Structure | RACC |      |
| ------------------------------------------------------------ | ------------ | --------------- | ---- | ---- |
| [Provably Robust Deep Learning via Adversarially Trained Smoothed Classifiers](https://arxiv.org/pdf/1906.04584.pdf) | Salman et al | ResNet-50       | 20%  |      |
| [Consistency Regularization for Certified Robustness of Smoothed Classifiers](https://arxiv.org/pdf/2006.04062.pdf) | Jeong et al  | ResNet-50       | 18%  |      |
| [MACER: Attack-free and Scalable Robust Training via Maximizing Certified Radius](https://arxiv.org/pdf/2001.02378.pdf) | Zhai et al   | ResNet-50       | 14%  |      |
| [Certified Adversarial Robustness via Randomized Smoothing](https://arxiv.org/pdf/1902.02918.pdf) | Cohen et al  | ResNet-50       | 12%  |      |

All above approaches use Randomized Smoothing (Cohen et al) to derive certification, with wrong probability at 0.1%.

#### L-Infty

##### eps=1/255

| Defense                                                      | Author       | Model Structure  | RACC  |                                                              |
| ------------------------------------------------------------ | ------------ | ---------------- | ----- | ------------------------------------------------------------ |
| [Provably Robust Deep Learning via Adversarially Trained Smoothed Classifiers](https://arxiv.org/pdf/1906.04584.pdf) | Salman et al | ResNet-50        | 36.8% | transformed from L-2 robustness; wrong prob. 0.001           |
| [Certified Adversarial Robustness via Randomized Smoothing](https://arxiv.org/pdf/1902.02918.pdf) | Cohen et al  | ResNet-50        | 28.6% | transformed from L-2 robustness by Salman et al; wrong prob. 0.001 |
| [On the Effectiveness of Interval Bound Propagation for Training Verifiably Robust Models](https://arxiv.org/pdf/1810.12715.pdf) | Gowal et al  | WideResNet-10-10 | 6.13% | Dataset downscaled to 64 x 64                                |



##### eps=1.785/255

| Defense                                                      | Author     | Model Structure | RACC  |                                |
| ------------------------------------------------------------ | ---------- | --------------- | ----- | ------------------------------ |
| [MixTrain: Scalable Training of Verifiably Robust Neural Networks](https://arxiv.org/pdf/1811.02625.pdf) | Wang et al | ResNet          | 19.4% |                                |
| [Scaling provable adversarial defenses](https://arxiv.org/pdf/1805.12514.pdf) | Wong et al | ResNet          | 5.1%  | Run and reported by Wang et al |

In above table, the dataset is ImageNet-200 rather than ImageNet-1000 in other tables.

### CIFAR-10

All input images have three channels; 32 x 32 x 3 size; each pixel is in range [0, 255].

#### L2

##### eps=0.14

| Defense                                                      | Author        | Model Structure | RACC   |                                        |
| ------------------------------------------------------------ | ------------- | --------------- | ------ | -------------------------------------- |
| [Scaling provable adversarial defenses](https://arxiv.org/pdf/1805.12514.pdf) | Wong et al    | Resnet          | 51.96% | 36/255; transformed from L-infty 2/255 |
| [Certified Robustness to Adversarial Examples with Differential Privacy](https://arxiv.org/pdf/1802.03471.pdf) | Lecuyer et al | Resnet          | 40%    |                                        |
| (Verification) [Efficient Neural Network Robustness Certification with General Activation Functions](https://papers.nips.cc/paper/7742-efficient-neural-network-robustness-certification-with-general-activation-functions.pdf) | Zhang et al   | ResNet-20       | 0%     | Reported by Cohen et al                |



##### eps=0.25

| Defense                                                      | Author       | Model Structure | RACC  |                          |
| ------------------------------------------------------------ | ------------ | --------------- | ----- | ------------------------ |
| [Provably Robust Deep Learning via Adversarially Trained Smoothed Classifiers](https://arxiv.org/pdf/1906.04584.pdf) | Salmon et al | ResNet-110      | 82%   |                          |
| [Unlabeled Data Improves Adversarial Robustness](https://arxiv.org/pdf/1905.13736.pdf) | Carmon et al | ResNet 28-10    | 72%   | interpolated from Fig. 1 |
| [MACER: Attack-free and Scalable Robust Training via Maximizing Certified Radius](https://arxiv.org/pdf/2001.02378.pdf) | Zhai et al   | ResNet-110      | 71%   |                          |
| [Consistency Regularization for Certified Robustness of Smoothed Classifiers](https://arxiv.org/pdf/2006.04062.pdf) | Jeong et al  | ResNet-110      | 67.5% |                          |
| [Certified Adversarial Robustness via Randomized Smoothing](https://arxiv.org/pdf/1902.02918.pdf) | Cohen et al  | ResNet-110      | 61%   |                          |
| (Verification) [Efficient Neural Network Robustness Certification with General Activation Functions](https://papers.nips.cc/paper/7742-efficient-neural-network-robustness-certification-with-general-activation-functions.pdf) | Zhang et al  | ResNet-20       | 0%    | Reported by Cohen et al  |

All above approaches use Randomized Smoothing (Cohen et al) to derive certification, with wrong probability at 0.1%.

##### eps=0.5

| Defense                                                      | Author       | Model Structure | RACC  |                          |
| ------------------------------------------------------------ | ------------ | --------------- | ----- | ------------------------ |
| [Provably Robust Deep Learning via Adversarially Trained Smoothed Classifiers](https://arxiv.org/pdf/1906.04584.pdf) | Salmon et al | ResNet-110      | 65%   |                          |
| [Unlabeled Data Improves Adversarial Robustness](https://arxiv.org/pdf/1905.13736.pdf) | Carmon et al | ResNet 28-10    | 61%   | interpolated from Fig. 1 |
| [MACER: Attack-free and Scalable Robust Training via Maximizing Certified Radius](https://arxiv.org/pdf/2001.02378.pdf) | Zhai et al   | ResNet-110      | 59%   |                          |
| [Consistency Regularization for Certified Robustness of Smoothed Classifiers](https://arxiv.org/pdf/2006.04062.pdf) | Jeong et al  | ResNet-110      | 57.7% |                          |
| [Certified Adversarial Robustness via Randomized Smoothing](https://arxiv.org/pdf/1902.02918.pdf]) | Cohen et al  | ResNet-110      | 43%   |                          |

All above approaches use Randomized Smoothing (Cohen et al) to derive certification, with wrong probability at 0.1%.

##### eps=1.0

| Defense                                                      | Author       | Model Structure | RACC  |      |
| ------------------------------------------------------------ | ------------ | --------------- | ----- | ---- |
| [Provably Robust Deep Learning via Adversarially Trained Smoothed Classifiers](https://arxiv.org/pdf/1906.04584.pdf) | Salmon et al | ResNet-110      | 39%   |      |
| [MACER: Attack-free and Scalable Robust Training via Maximizing Certified Radius](https://arxiv.org/pdf/2001.02378.pdf) | Zhai et al   | ResNet-110      | 38%   |      |
| [Consistency Regularization for Certified Robustness of Smoothed Classifiers](https://arxiv.org/pdf/2006.04062.pdf) | Jeong et al  | ResNet-110      | 37.8% |      |
| [Certified Adversarial Robustness via Randomized Smoothing](https://arxiv.org/pdf/1902.02918.pdf) | Cohen et al  | ResNet-110      | 22%   |      |

All above approaches use Randomized Smoothing (Cohen et al) to derive certification, with wrong probability at 0.1%.

##### eps=1.5

| Defense                                                      | Author       | Model Structure | RACC  |      |
| ------------------------------------------------------------ | ------------ | --------------- | ----- | ---- |
| [Provably Robust Deep Learning via Adversarially Trained Smoothed Classifiers](https://arxiv.org/pdf/1906.04584.pdf) | Salmon et al | ResNet-110      | 32%   |      |
| [Consistency Regularization for Certified Robustness of Smoothed Classifiers](https://arxiv.org/pdf/2006.04062.pdf) | Jeong et al  | ResNet-110      | 27.0% |      |
| [MACER: Attack-free and Scalable Robust Training via Maximizing Certified Radius](https://arxiv.org/pdf/2001.02378.pdf) | Zhai et al   | ResNet-110      | 25%   |      |
| [Certified Adversarial Robustness via Randomized Smoothing](https://arxiv.org/pdf/1902.02918.pdf) | Cohen et al  | ResNet-110      | 14%   |      |

All above approaches use Randomized Smoothing (Cohen et al) to derive certification, with wrong probability at 0.1%.

#### L Infty

##### eps=2/255

| Defense/Verification                                         | Author          | Model Structure | RACC        |                                                              |
| ------------------------------------------------------------ | --------------- | --------------- | ----------- | ------------------------------------------------------------ |
| [Provably Robust Deep Learning via Adversarially Trained Smoothed Classifiers](https://arxiv.org/pdf/1906.04584.pdf) | Salmon et al    | ResNet-110      | 68.2%       | transformed from L-2 robustness; wrong prob. 0.1%            |
| (Verification) [Efficient Neural Network Verification with Exactness Characterization](http://auai.org/uai2019/proceedings/papers/164.pdf) | Dvijotham et al | Small CNN       | 65.4%       |                                                              |
| [Unlabeled Data Improves Adversarial Robustness](https://arxiv.org/pdf/1905.13736.pdf) | Carmon et al    | ResNet 28-10    | 63.8% ± 0.5 | wrong prob. 0.1%                                             |
| [Adversarial Training and Provable Defenses: Bridging the Gap](https://files.sri.inf.ethz.ch/website/papers/iclr2020-colt.pdf) | Baunovic et al  | 4-layer CNN     | 60.5%       |                                                              |
| [Robustra: Training Provable Robust Neural Networks over Reference Adversarial Space](https://www.ijcai.org/proceedings/2019/654) | Li et al        | CNN             | 56.32%      |                                                              |
| [Towards Stable and Efficient Training of Verifiably Robust Neural Networks](https://arxiv.org/pdf/1906.06316.pdf) | Zhang et al     | small CNN       | 53.97%      | pick the best number                                         |
| [Scaling provable adversarial defenses](https://arxiv.org/pdf/1902.02918.pdf) | Wong et al      | Resnet          | 53.89%      |                                                              |
| [Differentiable Abstract Interpretation for Provably Robust Neural Networks](http://proceedings.mlr.press/v80/mirman18b/mirman18b.pdf) | Mirman et al    | Residual        | 52.2%       | ~1.785/255, only evaluated from 500 samples among all 10,000 |
| [MixTrain: Scalable Training of Verifiably Robust Neural Networks](https://arxiv.org/pdf/1811.02625.pdf) | Wang et al      | Resnet          | 50.4%       |                                                              |
| (Verification)[Evaluating Robustness of Neural Networks with Mixed Integer Programming](https://arxiv.org/pdf/1711.07356.pdf) | Tjeng et al     | CNN             | 50.20%      | MILP verification on Wong et al. model                       |
| [On the Effectiveness of Interval Bound Propagation for Training Verifiably Robust Models](https://arxiv.org/pdf/1810.12715.pdf) | Gowal et al     | CNN             | 50.02%      |                                                              |
| [Training for Faster Adversarial Robustness Verification via Inducing ReLU Stability](https://arxiv.org/pdf/1809.03008.pdf) | Xiao et al      | CNN             | 45.93%      |                                                              |
| (Verification) [Beyond the Single Neuron Convex Barrier for Neural Network Certification](https://www.sri.inf.ethz.ch/publications/singh2019krelu) | Singh et al     | ConvBig         | 45.9%       | evaluated on first 1,000 images                              |
| (Verification) [An Abstract Domain for Certifying Neural Networks](https://files.sri.inf.ethz.ch/website/papers/DeepPoly.pdf) | Singh et al     | CNN             | 40%         | ~2.04/255, only evaluated from 100 samples among all 10,000  |
| (Verification)[A Dual Approach to Scalable Verification of Deep Networks](http://auai.org/uai2018/proceedings/papers/204.pdf) | Dvijotham et al | CNN             | 20%         | LP-Dual verification on Uesato et al. model; interpolated from Fig. 2(a) |



##### eps=8/255

| Defense/Verification                                         | Author                | Model Structure     | RACC   |                                                              |
| ------------------------------------------------------------ | --------------------- | ------------------- | ------ | ------------------------------------------------------------ |
| [Fast and Stable Interval Bounds Propagation for Training Verifiably Robust Models](https://arxiv.org/pdf/1906.00628.pdf) | Morawiecki et al      | Small CNN           | 39.88% |                                                              |
| [Differentiable Abstract Interpretation for Provably Robust Neural Networks](http://proceedings.mlr.press/v80/mirman18b/mirman18b.pdf) | Mirman et al          | Small CNN           | 37.4%  | ~7.65/255, only evaluated from 500 samples among all 10,000  |
| [Towards Stable and Efficient Training of Verifiably Robust Neural Networks](https://arxiv.org/pdf/1906.06316.pdf) | Zhang et al           | large CNN(DM-large) | 33.06% | pick the best number                                         |
| [On the Effectiveness of Interval Bound Propagation for Training Verifiably Robust Models](https://arxiv.org/pdf/1810.12715.pdf) | Gowal et al           | CNN                 | 32.04% | Practically reproducible verified error is about 28% - 29% according to Zhang et al |
| [Adversarial Training and Provable Defenses: Bridging the Gap](https://files.sri.inf.ethz.ch/website/papers/iclr2020-colt.pdf) | Baunovic et al        | 4-layer CNN         | 27.5%  |                                                              |
| [Training Verified Learners with Learned Verifiers](https://arxiv.org/pdf/1805.10265.pdf) | Dvijotham et al       | Predictor-Verifier  | 26.67% |                                                              |
| [Provably Robust Boosted Decision Stumps and Trees against Adversarial Attacks](https://arxiv.org/pdf/1906.03526.pdf) | Andriushchenko & Hein | Boosted trees       | 25.31% |                                                              |
| [Robustra: Training Provable Robust Neural Networks over Reference Adversarial Space](https://www.ijcai.org/proceedings/2019/654) | Li et al              | CNN                 | 25.13% |                                                              |
| (Verification) [Beyond the Single Neuron Convex Barrier for Neural Network Certification](https://www.sri.inf.ethz.ch/publications/singh2019krelu) | Singh et al           | ResNet              | 24.5%  | Evaluated on first 1,000 images                              |
| [A Provable Defense for Deep Residual Networks](https://arxiv.org/pdf/1903.12519.pdf) | Mirman et al          | ResNet-Tiny         | 23.2%  |                                                              |
| [Evaluating Robustness of Neural Networks with Mixed Integer Programming](https://arxiv.org/pdf/1711.07356.pdf) | Tjeng et al           | CNN                 | 22.40% |                                                              |
| [Scaling provable adversarial defenses](https://arxiv.org/pdf/1902.02918.pdf) | Wong et al            | Resnet              | 21.78% |                                                              |
| (Verification) [Boosting Robustness Certification of Neural Networks](https://files.sri.inf.ethz.ch/website/papers/DeepZ.pdf) | Singh et al           | Small CNN           | 21%    | ~7.65/255, only evaluated from 500 samples among all 10,000  |
| [Training for Faster Adversarial Robustness Verification via Inducing ReLU Stability](https://arxiv.org/pdf/1809.03008.pdf) | Xiao et al            | CNN                 | 20.27% |                                                              |
| (Verification) [A Dual Approach to Scalable Verification of Deep Networks](http://auai.org/uai2018/proceedings/papers/204.pdf) | Dvijotham et al       | CNN                 | 0%     | LP-Dual verification on Uesato et al. model; interpolated from Fig. 2(a) |



### MNIST

All input images are grayscale; 28 x 28 size; each pixel is in range [0, 1].

#### L2

##### eps=1.58

eps=1.58 is transformed from L-infty eps=0.1.

| Defense/Verification                                         | Author           | Model Structure    | RACC   |                             |
| ------------------------------------------------------------ | ---------------- | ------------------ | ------ | --------------------------- |
| [Scaling provable adversarial defenses](https://arxiv.org/pdf/1902.02918.pdf) | Wong et al       | Small CNN          | 88.14% |                             |
| [Consistency Regularization for Certified Robustness of Smoothed Classifiers](https://arxiv.org/pdf/2006.04062.pdf) | Jeong et al      | LeNet              | 81.5%  | slightly smaller radius 1.5 |
| [Second-Order Provable Defenses against Adversarial Attacks](https://arxiv.org/pdf/2006.00731.pdf) | Singla and Feizi | 2x[1024], softplus | 69.79% |                             |



#### L-Infty

##### eps=0.1

| Defense/Verification                                         | Author            | Model Structure    | RACC   |                                                  |
| ------------------------------------------------------------ | ----------------- | ------------------ | ------ | ------------------------------------------------ |
| [Robustra: Training Provable Robust Neural Networks over Reference Adversarial Space](https://www.ijcai.org/proceedings/2019/0654.pdf) | Li et al          | large CNN          | 97.91% |                                                  |
| [On the Effectiveness of Interval Bound Propagation for Training Verifiably Robust Models](https://arxiv.org/pdf/1810.12715.pdf) | Gowal et al       | CNN                | 97.77% |                                                  |
| [Towards Stable and Efficient Training of Verifiably Robust Neural Networks](https://arxiv.org/pdf/1906.06316.pdf) | Zhang et al       | small CNN          | 97.76% |                                                  |
| (Verification) [Evaluating Robustness of Neural Networks with Mixed Integer Programming](https://arxiv.org/abs/1711.07356) | Tjeng et al       | large CNN          | 97.26% |                                                  |
| [Adversarial Training and Provable Defenses: Bridging the Gap](https://files.sri.inf.ethz.ch/website/papers/iclr2020-colt.pdf) | Baunovic et al    | 3-layer CNN        | 97.1%  |                                                  |
| [MixTrain: Scalable Training of Verifiably Robust Neural Networks](https://arxiv.org/pdf/1811.02625.pdf) | Wang et al        | small CNN          | 97.1%  |                                                  |
| (Verification) [Boosting Robustness Certification of Neural Networks](https://files.sri.inf.ethz.ch/website/papers/DeepZ.pdf) | Singh et al       | ConvSuper          | 97%    | Only evaluated from 100 samples among all 10,000 |
| [Differentiable Abstract Interpretation for Provably Robust Neural Networks](http://proceedings.mlr.press/v80/mirman18b/mirman18b.pdf) | Mirman et al      | big CNN            | 96.6%  | Only evaluated from 500 samples among all 10,000 |
| [Scaling Provable Adversarial Defenses](https://arxiv.org/pdf/1805.12514.pdf) | Wong et al        | large CNN          | 96.33% |                                                  |
| [Training Verified Learners with Learned Verifiers](https://arxiv.org/pdf/1805.10265.pdf) | Dvijotham et al   | Predictor-Verifier | 95.56% |                                                  |
| [Provable Defenses against Adversarial Examples via the Convex Outer Adversarial Polytope](https://arxiv.org/pdf/1711.00851.pdf) | Wong et al        | CNN                | 94.18% |                                                  |
| (Verification) [Efficient Neural Network Verification with Exactness Characterization](http://auai.org/uai2019/proceedings/papers/164.pdf) | Dvijotham et al   | Grad-NN            | 83.68% |                                                  |
| (Verification) [A Convex Relaxation Barrier to Tight Robustness Verification of Neural Networks](https://arxiv.org/pdf/1902.08722.pdf) | Salman et al      | CNN                | 82.76% |                                                  |
| (Verification) [Semidefinite relaxations for certifying robustness to adversarial examples](https://arxiv.org/pdf/1811.01057.pdf) | Raghunathan et al | small NN           | 82%    |                                                  |
| [Certified Defenses against Adversarial Examples](https://arxiv.org/pdf/1801.09344.pdf) | Raghunathan et al | 2-layer NN         | 65%    |                                                  |



##### eps=0.3

| Defense/Verification                                         | Author                | Model Structure | RACC   |                                 |
| ------------------------------------------------------------ | --------------------- | --------------- | ------ | ------------------------------- |
| [Towards Stable and Efficient Training of Verifiably Robust Neural Networks](https://arxiv.org/pdf/1906.06316.pdf) | Zhang et al           | large CNN       | 92.98% | pick the best number            |
| [On the Effectiveness of Interval Bound Propagation for Training Verifiably Robust Models](https://arxiv.org/pdf/1810.12715.pdf) | Gowal et al           | CNN             | 91.95% |                                 |
| [Provably Robust Boosted Decision Stumps and Trees against Adversarial Attacks](https://arxiv.org/pdf/1906.03526.pdf) | Andriushchenko & Hein | Boosted trees   | 87.54% |                                 |
| [Adversarial Training and Provable Defenses: Bridging the Gap](https://files.sri.inf.ethz.ch/website/papers/iclr2020-colt.pdf) | Baunovic et al        | 3-layer CNN     | 85.7%  |                                 |
| [Robustra: Training Provable Robust Neural Networks over Reference Adversarial Space](https://www.ijcai.org/proceedings/2019/0654.pdf) | Li et al              | small CNN       | 83.09% |                                 |
| (Verification) [Evaluating Robustness of Neural Networks with Mixed Integer Programming](https://arxiv.org/abs/1711.07356) | Tjeng et al           | large CNN       | 75.81% |                                 |
| (Verification) [Beyond the Single Neuron Convex Barrier for Neural Network Certification](https://www.sri.inf.ethz.ch/publications/singh2019krelu) | Singh et al           | ConvBig         | 73.6%  | evaluated on first 1,000 images |
| [MixTrain: Scalable Training of Verifiably Robust Neural Networks](https://arxiv.org/pdf/1811.02625.pdf) | Wang et al            | small CNN       | 60.1%  |                                 |
| [Scaling Provable Adversarial Defenses](https://arxiv.org/pdf/1805.12514.pdf) | Wong et al            | small CNN       | 56.90% |                                 |
| (Verification) [A Convex Relaxation Barrier to Tight Robustness Verification of Neural Networks](https://arxiv.org/pdf/1902.08722.pdf) | Salman et al          | CNN             | 39.83% |                                 |



##### eps=0.4

| Defense/Verification                                         | Author           | Model Structure | RACC   |                      |
| ------------------------------------------------------------ | ---------------- | --------------- | ------ | -------------------- |
| [Towards Stable and Efficient Training of Verifiably Robust Neural Networks](https://arxiv.org/pdf/1906.06316.pdf) | Zhang et al      | large CNN       | 87.94% | pick the best number |
| [On the Effectiveness of Interval Bound Propagation for Training Verifiably Robust Models](https://arxiv.org/pdf/1810.12715.pdf) | Gowal et al      | CNN             | 85.12% |                      |
| [Fast and Stable Interval Bounds Propagation for Training Verifiably Robust Models](https://arxiv.org/pdf/1906.00628.pdf) | Morawiecki et al | large CNN       | 84.42% |                      |
| (Verification) [Evaluating Robustness of Neural Networks with Mixed Integer Programming](https://arxiv.org/abs/1711.07356) | Tjeng et al      | small CNN       | 51.02% |                      |





### SVHN

The image size is 32 x 32 x 3 (3-channel in color). Pixel colors in [0, 255]. When calculating eps, these values are rescaled to [0, 1].

#### L2

##### eps=0.1

| Defense/Verification                                         | Author        | Model Structure | RACC |                              |
| ------------------------------------------------------------ | ------------- | --------------- | ---- | ---------------------------- |
| [Certified Adversarial Robustness via Randomized Smoothing](https://arxiv.org/pdf/1902.02918.pdf) | Cohen et al   | Resnet-20       | ~95% | Interpolate from Cohen et al |
| [Lipschitz-Margin Training: Scalable Certification of Perturbation Invariance for Deep Neural Networks](https://arxiv.org/pdf/1802.04034.pdf) | Tsuzuku et al | Resnet-20       | 0%   | Interpolate from Cohen et al |



##### eps=0.2

| Defense/Verification                                         | Author        | Model Structure | RACC |                              |
| ------------------------------------------------------------ | ------------- | --------------- | ---- | ---------------------------- |
| [Certified Adversarial Robustness via Randomized Smoothing](https://arxiv.org/pdf/1902.02918.pdf) | Cohen et al   | Resnet-20       | ~88% | Interpolate from Cohen et al |
| [Lipschitz-Margin Training: Scalable Certification of Perturbation Invariance for Deep Neural Networks](https://arxiv.org/pdf/1802.04034.pdf) | Tsuzuku et al | Resnet-20       | 0%   | Interpolate from Cohen et al |



#### L-Infty

##### eps=0.01

| Defense/Verification                                         | Author          | Model Structure    | RACC   |      |
| ------------------------------------------------------------ | --------------- | ------------------ | ------ | ---- |
| [Adversarial Training and Provable Defenses: Bridging the Gap](https://files.sri.inf.ethz.ch/website/papers/iclr2020-colt.pdf) | Baunovic et al  | 3-layer CNN        | 70.2%  |      |
| [Training Verified Learners with Learned Verifiers](https://arxiv.org/pdf/1805.10265.pdf) | Dvijotham et al | Predictor-Verifier | 62.44% |      |
| [On the Effectiveness of Interval Bound Propagation for Training Verifiably Robust Models](https://arxiv.org/pdf/1810.12715.pdf) | Gowal et al     | CNN                | 62.40% |      |
| [Provable Defenses against Adversarial Examples via the Convex Outer Adversarial Polytope](https://arxiv.org/pdf/1711.00851.pdf) | Wong et al      | CNN                | 59.33% |      |
| [Differentiable Abstract Interpretation for Provably Robust Neural Networks](http://proceedings.mlr.press/v80/mirman18b/mirman18b.pdf) | Mirman et al    | small CNN          | 11.0%  |      |



##### eps=8/255

| Defense/Verification                                         | Author           | Model Structure | RACC   |                            |
| ------------------------------------------------------------ | ---------------- | --------------- | ------ | -------------------------- |
| [On the Effectiveness of Interval Bound Propagation for Training Verifiably Robust Models](https://arxiv.org/pdf/1810.12715.pdf) | Gowal et al      | large CNN       | 47.63% | Report by Morawiecki et al |
| [Fast and Stable Interval Bounds Propagation for Training Verifiably Robust Models](https://arxiv.org/pdf/1906.00628.pdf) | Morawiecki et al | small CNN       | 46.03% |                            |



### Fashion-MNIST

This is a MNIST-like dataset. Images are 28 x 28 and grayscale. Values are in [0, 1].

#### L-Infty

##### eps=0.1

| Defense/Verification                                         | Author                | Model Structure | RACC   |                                               |
| ------------------------------------------------------------ | --------------------- | --------------- | ------ | --------------------------------------------- |
| [Towards Stable and Efficient Training of Verifiably Robust Neural Networks](https://arxiv.org/pdf/1906.06316.pdf) | Zhang et al           | large CNN       | 78.73% | pick the best number                          |
| [On the Effectiveness of Interval Bound Propagation for Training Verifiably Robust Models](https://arxiv.org/pdf/1810.12715.pdf) | Gowal et al           | large CNN       | 77.63% | pick the best number, reported by Zhang et al |
| [Provably Robust Boosted Decision Stumps and Trees against Adversarial Attacks](https://arxiv.org/pdf/1906.03526.pdf) | Andriushchenko & Hein | Boosted trees   | 76.83% |                                               |
| [Provable Defenses against Adversarial Examples via the Convex Outer Adversarial Polytope](https://arxiv.org/pdf/1711.00851.pdf) | Wong et al            | CNN             | 65.47% |                                               |



**\***. Within one dataset, L-2 and L-Infty balls are mutually transformable. After transformation, a corresponding tight bound may exist but not listed.

**Notes**:

1. Some papers use rarely-used epsilon to report their results, which may increase comparison difficulty. Some papers use epsilon after regularization instead of raw one, which may also induce confusion.

   We would suggest to adapt common evaluation epsilon values and settings.

2. Instead of evaluating on above benchmarks and reporting the robust accuracy, some papers tend to report average robust radius. We will add comparison table for such metric later.

4. Besides the on-the-board results, all these papers have their own unique takeaways. For interested reader and stackholders, we recommend to not only value the approach with higher numbers, but also dig into their technical meat.

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



## Taxnomy Tree

![Taxonomy Tree](https://github.com/AI-secure/Provable-Training-and-Verification-Approaches-Towards-Robust-Neural-Networks/blob/master/taxonomy_tree.jpg)

---

Maintained by Linyi.

Last updated: Sept 15, 2020
