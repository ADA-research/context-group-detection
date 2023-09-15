# Group Detection from Spatiotemporal Data using Social Context

[//]: # (todo add introduction)

## T-DANTE

In this section the architecture of our method is going to be introduced. The first thing to be discussed is the
approach
to learn the affinities between agents in a scene through the use of a Deep Neural Network (DNN). Since our network is
based on DANTE [[5]](#5) and we applied LSTM/GRU layers to include temporal features of spatiotemporal data, we
name our model T-DANTE. Figure 1 and Figure 2 give a visual
representation for the first and second part of our framework, respectively.

![trajectories to affinity graph](models/pngs/trajectories%20to%20affinity%20graph.png)
*Figure 1: Trajectories to affinity graph*

![graph community detection](models/pngs/graph%20community%20detection.png)
*Figure 2: Affinity graph community detection*

T-DANTE is a deep neural network (DNN) that predicts the weights for each of the edges in the
affinity graph (Figure 1). T-DANTE is structured to exploit two types of information: local spatial
information from the two nodes (individuals) connected to an edge of interest, and global spatial information from other
nearby people, who form the social context of the pair of interest. This is the idea introduced in DANTE by Swofford et
al. [[5]](#5). T-DANTE advances this idea by using LSTMs/GRUs that are making it possible to use data of multiple
past timeframes to decide the affinity score between two agents and not use only spatial features.

## Datasets

### Pedestrian Datasets

<img src="datasets/ETH/seq_eth/reference.png" alt="eth reference image" style="width:200px;"/>
<img src="datasets/ETH/seq_hotel/reference.png" alt="hotel reference image" style="width:200px;"/>
<img src="datasets/UCY/zara01/reference.png" alt="zara01/02 reference image" style="width:200px;"/>
<img src="datasets/UCY/students03/reference.png" alt="students03 reference image" style="width:200px;"/>

The chosen datasets for our experiments are 2 public pedestrian datasets containing multiple experiments with
information about group relationships. The first is BIWI Walking Pedestrian Dataset from Pellegrini et al. [[1]](#1)
which contains two experiments *ETH* and *Hotel*. The second is UCY dataset from
Lerner et al. [[2]](#2), which includes 3 experiments, namely *zara01*, *zara02* and
*students03*, with group information about the subjects. These datasets can be found at
[OpenTraj](https://github.com/crowdbotp/OpenTraj) and are commonly used as benchmarks for
group detection on spatiotemporal data.
The data of the aforementioned experiments consist of the location and the velocity of each agent for multiple time
frames. The ground truth of the agent groups is also included in the datasets.

### Simulation Datasets

<img src="datasets/simulation/sim_10_3_2_4/simulation.gif" alt="simulation example gif" style="width:400px;"/>

In addition to experimenting with real world datasets, spring simulation data have been used in our experiments.
Simulation data are generated, so the ground truth is known and an infinite amount of data can be produced in order to
have enough data for our model to be trained. The spring simulation data that have been used were proposed by Kipf et
al. [[3]](#3) and were enriched by Nasri et al. [[4]](#4) with particle group information. The basic idea
is that a number of particles move in a 2-D space, simulating the concept of particles moving along with each other and
affecting the trajectory of each other. Locations and velocities of the particles are part of the generated data as well
as the communities that they belong to.

### Dataset Preprocessing

[//]: # (todo add how to run files)

## Baselines

1. DANTE

- Swofford et al. [[5]](#5) presented a data-driven approach to detect conversational groups. Their approach introduced
  a novel Deep Affinity Network (DANTE) to predict the likelihood that two agents in the same scene can be part of the
  same conversational group, considering their social context. In more detail, DANTE is a neural network that takes the
  location and head orientation data of a single frame scene and tries to learn the pairwise affinities between the
  agents by identifying their spacial arrangements. The predicted results for all agent pairs in the scene are then used
  by a clustering algorithm to identify groups of various sizes. This pipeline was also used to test interaction
  scenarios between a robot and humans. Instead of relying on head orientation, velocity data from our datasets is
  utilized for this baseline. The lack of temporal considerations for the problem is expected to make it unable to
  perform as well as our model.

2. WavenetNRI

- Nasri et al. [[4]](#4) used an NRI [[3]](#3) adaptation to perform group detection in spatiotemporal data. The model
  consists of a Graph Neural Network (GNN) encoder transformed by applying a Residual Dilated Causal Convolutional Block
  inspired by Wavenet architecture [[6]](#6). This work includes both supervised and unsupervised training.
  Louvain community detection algorithm is used to find the clusters of the interaction graphs formed by the predictions
  of the model. For our experiments we have used the supervised trained version as a baseline. This model uses whole
  scenes as samples, something that is different from our approach that uses only a specific amount of surrounding
  agents to predict the affinities.

## Results

[//]: # (todo add results and how to run experiments)

## References

<a id="1">[1]</a>
Stefano Pellegrini, Andreas Ess, Konrad Schindler, and Luc Van Gool. You’ll never walk alone: Modeling social behavior
for multi-target tracking. In 2009 IEEE 12th International Conference on Computer Vision, pages 261–268. IEEE, 2009.

<a id="2">[2]</a>
Alon Lerner, Yiorgos Chrysanthou, and Dani Lischinski. Crowds by example. In Computer graphics forum, volume 26, pages
655–664. Wiley Online Library, 2007. 30

<a id="3">[3]</a>
Thomas Kipf, Ethan Fetaya, Kuan-Chieh Wang, Max Welling, and Richard Zemel. Neural relational inference for interacting
systems. In Jennifer Dy and Andreas Krause, editors, Proceedings of the 35th International Conference on Machine
Learning, volume 80 of Proceedings of Machine Learning Research, pages 2688–2697. PMLR, 10–15 Jul 2018.

<a id="4">[4]</a>
Maedeh Nasri, Zhizhou Fang, Mitra Baratchi, Gwenn Englebienne, Shenghui Wang, Alexander Koutamanis, and Carolien Rieffe.
A gnn-based architecture for group detection from spatio-temporal trajectory data. In Bruno Cr ́emilleux, Sibylle Hess,
and Siegfried Nijssen, editors, Advances in Intelligent Data Analysis XXI, pages 327–339, Cham, 2023. Springer Nature
Switzerland.

<a id="5">[5]</a>
Mason Swofford, John Peruzzi, Nathan Tsoi, Sydney Thompson, Roberto Mart ́ın-Mart ́ın, Silvio Savarese, and Marynel V
́azquez. Improving social awareness through dante: Deep affinity network for clustering conversational interactants.
Proc. ACM Hum.-Comput. Interact., 4(CSCW1), may 2020.

<a id="6">[6]</a>
A ̈aron van den Oord, Sander Dieleman, Heiga Zen, Karen Simonyan, Oriol Vinyals, Alexander Graves, Nal Kalchbrenner,
Andrew Senior, and Koray Kavukcuoglu. Wavenet: A generative model for raw audio. In Arxiv, 2016.
