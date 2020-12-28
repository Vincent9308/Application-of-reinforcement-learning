# Application-of-reinforcement-learning
Classical reinforcement learning application scenarios and state-of-the-art methods.


介绍现阶段强化学习可应用的几种典型场景，以及目前在该场景中体现出最高水平的方法。

## 目录

* [开发与训练](#开发与训练)
    * [深度学习框架](#深度学习框架)
    * [分布式训练](#分布式训练)
* [应用](#应用)
    * [游戏](#游戏)
      * [无博弈](#无博弈游戏)
      * [博弈类](#博弈类游戏)
        * [模型已知](#模型已知)
        * [模型未知](#模型未知)
    * [推荐系统](#推荐系统)
    * [机器人](#机器人)
    * [智能交通](#智能交通)
    * [计算机系统](#计算机系统)
    * [自动驾驶](#自动驾驶)
* [部署](#部署)
* [应用设计流程](#设计流程)


## 开发与训练

### 深度学习框架

| 框架                                                         | 简介                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [PyTorch](https://pytorch.org/) | Facebook开源的深度学习框架 |
| [TensorFlow](https://www.tensorflow.org/) | Google开源的深度学习框架 |
| [飞桨](https://www.paddlepaddle.org.cn/) | 百度开源的深度学习框架 |


### 分布式训练

| 框架                                                         | 简介                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [Ray](https://ray.io/) | UC Berkeley RISELab 出品的机器学习分布式框架 |

## 应用

### 游戏

#### 无博弈游戏

| 方法                                                         | 简介                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [Agent57](https://deepmind.com/blog/article/Agent57-Outperforming-the-human-Atari-benchmark) | DeepMind 构建了一个名为 Agent57 的智能体，该智能体在街机学习环境（Arcade Learning Environment，ALE）数据集所有 57 个雅达利游戏中实现了超越人类的表现。 |


#### 博弈类游戏

##### 模型已知
| 方法                                                         | 简介                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [AlphaZero](https://deepmind.com/blog/article/alphazero-shedding-new-light-grand-games-chess-shogi-and-go) | 从零开始，靠纯自我博弈掌握棋类游戏 |
| [ELF](https://github.com/pytorch/ELF) | Facebook 对 alphazero 的开源实现 |
| [MuZero](https://deepmind.com/blog/article/muzero-mastering-go-chess-shogi-and-atari-without-rules) | Model-based 方法的里程碑之作，无需知晓规则即可掌握围棋、国际象棋、shogi 和 Atari，并达到匹敌 Alphazero 的水平。  |


##### 模型未知

| 方法                                                         | 简介                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [AlphaStar](https://deepmind.com/blog/article/alphastar-mastering-real-time-strategy-game-starcraft-ii) | 提出了一个端到端的训练方法，最后训练出来的 AlphaStar 打败了职业星际玩家，超过了 99.8% 的人类玩家。在训练方法上，使用人类数据和智能体对弈数据，使用了多智能体强化学习方法；特别地，设计了若干策略池（league）来连续地学习策略和反制策略。 |
| [OpenAI Five](https://openai.com/blog/openai-five/) | OpenAI Five 在2019年4月13日击败了 Dota2 世界冠军战队OG |
| [JueWu](https://arxiv.org/pdf/2011.12692v1.pdf) | 腾讯AI「绝悟」，可达职业战队水平 |

### 推荐系统

| 方法                                                         | 简介                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [Youtube](https://arxiv.org/pdf/1812.02353.pdf) | 谷歌研究人员将强化学习中经典的REINFORCE算法应用于youtube的视频推荐中 |

### 机器人

| 方法                                                         | 简介                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [Automatic Domain Randomization](https://openai.com/blog/solving-rubiks-cube/) | OpenAI成功地在模拟器训练机械手臂，在真实环境中部署，完成魔方控制任务 |
| [RoboImitation](https://xbpeng.github.io/projects/Robotic_Imitation/index.html) | Learning Agile Robotic Locomotion Skills by Imitating Animals |

### 智能交通

| 方法                                                         | 简介                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [didi](https://www.kdd.org/kdd2019/accepted-papers/view/a-deep-value-network-based-approach-for-multi-driver-order-dispatching#!) | 提出了一种新的基于深度强化学习与半马尔科夫决策过程的智能派单应用，在同时考虑时间与空间的长期优化目标的基础上利用深度神经网络进行更准确有效的价值估计。通过系统的离线模拟实验以及在滴滴平台的在线AB实验证明，这种基于深度强化学习的派单算法相比现有最好的方法能进一步显著提升平台各项效率及用户体验。 |

### 计算机系统

| 方法                                                         | 简介                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [Park](http://people.csail.mit.edu/hongzi/content/publications/Park-NIPS19.pdf) | An Open Platform for Learning Augmented Computer Systems |



## 部署

| 工具                                                         | 简介                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [PyTorch Mobile](https://pytorch.org/mobile/home/) |  |
| [TensorFlow Lite](https://www.tensorflow.org/lite/) |  |
| [Paddle Lite](https://www.paddlepaddle.org.cn/paddle/paddlelite) |  |


## 设计流程

* 是否可以用强化学习的方法来解决（能不能）
* 是否适合用强化学习的方法来解决（要不要用，是否比传统方法效率高或达到的效果好）
* MDP建模
    * 状态
    * 动作
    * 奖励
* 根据任务的具体情况选取适合的强化学习训练算法
* 用来近似值函数或策略函数的神经网络结构设计
* 是否有必要结合监督学习进行网络初始化
* 是否有必要结合模仿学习进行伪奖励的设计
* 训练方式的设计，考虑：
    * 是否需要对手
        * elf play
    * 策略的鲁棒性
        * 联盟竞技场
    * 环境发生变化时，是否有必要将以前大规模训练的结果迁移过来，避免从头开始训练
        * 迁移学习
* 确定整个系统的运行流程