> **Temporal Imitation Learning in End-to-End Driving Models** <br>
> Maximilian Hilbert, [Andreas Geiger](https://www.cvlibs.net/) <be><br>
> [Slides, coming soon](LINK TO SLIDES)<br>
> [Thesis, coming soon](LINK TO Thesis)<br>
> This repo contains the code for the master's thesis **Temporal Imitation Learning in End-to-End Driving Models**, which is based on the Carla Garage Repository of our research group https://github.com/autonomousvision/carla_garage by [Bernhard Jaeger](https://kait0.github.io/)

## Contents
1. [Abstract](#abstract)
2. [Method](#method)
3. [Key Findings](#key_findings)
4. [Visualisations](#Visualisations)

## Abstract
This thesis aims to revisit known solutions to temporally infused end-to-end autonomous driving and update them to new architectures, output representations, and auxiliary tasks, as well as answering the question if temporal information is necessary to achieve perfect success scores on the Carla Nocrash Benchmark. Through rigorous closed-loop evaluation and ablation studies, we tested a variety of configurations, including different backbones, additional measurement inputs, auxiliary losses, and data augmentation techniques. Our experiments show that pre-trained video models are generally capable of improving downstream driving performance, attributing a part of the performance gap between the single-observation and multi-observation baselines to a lack of temporal scene understanding and not necessarily to the copycat problem, assumed by previous work in the field. Moreover, we demonstrate that our model (TimeFuser), based solely on single observations and therefore lacking any temporal context, is capable of achieving a perfect success rate on the Carla Nocrash Benchmark, challenging the implicit notion of previous work in the field, that temporal context is necessary for solving this benchmark.

## Method
The goal of this thesis is to build a testbed for reproducing known solutions to the copycat problem[1,2,3] as well as researching possible extensions concerning output representation, auxiliary tasks, temporally consistent augmentation strategies and video models as backbones and answer the question of whether temporal information is needed for completing the Nocrash Benchmark with a perfect success score.
<p align="center">
  <img src="assets/TimeFuser.png" alt="TimeFuser" width="500"/>
</p>

## Key Findings
## Finding One - Temporally consistent augmentation strategy
Training temporally infused end-to-end models are challenging due to the compounding errors problem during inference. To mitigate this issue one needs a temporally consistent (steering) augmentation strategy which i implemented using triangular perturbations[1]. One can see that this strategy greatly improves the driving task for all temporal and non-temporal baselines.
<table>
    <caption><strong>Carla NoCrash Benchmark[1] - Town02 (Test) - Dense Traffic (hardest) - Triangular Perturbation</strong></caption>
    <thead>
        <tr>
            <th>Baseline</th>
            <th>Aug.</th>
            <th>Timeouts &#x2193;</th>
            <th>Collisions &#x2193;</th>
            <th>Success &#x2191;</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan="2">ARP (temporal, two-stream)</td>
            <td>-</td>
            <td>0 &plusmn; 0</td>
            <td>47 &plusmn; 10</td>
            <td>53 &plusmn; 10</td>
        </tr>
        <tr>
            <td>&#x2713;</td>
            <td>3 &plusmn; 3</td>
            <td>14 &plusmn; 5</td>
            <td>83 &plusmn; 6</td>
        </tr>
        <tr>
            <td rowspan="2">BCOH (temporal, single-stream)</td>
            <td>-</td>
            <td>0 &plusmn; 0</td>
            <td>50 &plusmn; 9</td>
            <td>50 &plusmn; 9</td>
        </tr>
        <tr>
            <td>&#x2713;</td>
            <td>13 &plusmn; 9</td>
            <td>21 &plusmn; 8</td>
            <td>65 &plusmn; 7</td>
        </tr>
        <tr>
            <td rowspan="2">BCSO (non-temporal)</td>
            <td>-</td>
            <td>2 &plusmn; 3</td>
            <td>28 &plusmn; 7</td>
            <td>70 &plusmn; 7</td>
        </tr>
        <tr>
            <td>&#x2713;</td>
            <td>5 &plusmn; 8</td>
            <td>6 &plusmn; 5</td>
            <td><strong>89</strong> &plusmn; 11</td>
        </tr>
    </tbody>
</table>

## Finding Two - Temporal baselines take advantage of video models for efficient encoding
This table demonstrates that video models, pre-trained on action recognition, significantly enhance perception quality for the default BCOH baseline, thus improving downstream driving performance compared to the default ResNet backbone, which is pre-trained on object detection.

<table>
    <caption><strong>Carla NoCrash Benchmark - Town02 (Test) - Dense Traffic (hardest) - Backbone Ablation</strong></caption>
    <thead>
        <tr>
            <th>Baseline</th>
            <th>Backbone</th>
            <th>Timeouts &#x2193;</th>
            <th>Collisions &#x2193;</th>
            <th>Success &#x2191;</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan="3">ARP (temporal, two-stream)</td>
            <td>Resnet</td>
            <td>0 &plusmn; 0</td>
            <td>45 &plusmn; 7</td>
            <td>55 &plusmn; 7</td>
        </tr>
        <tr>
            <td>Swin</td>
            <td>5 &plusmn; 4</td>
            <td>39 &plusmn; 8</td>
            <td>56 &plusmn; 10</td>
        </tr>
        <tr>
            <td>X3D</td>
            <td>3 &plusmn; 3</td>
            <td>39 &plusmn; 5</td>
            <td>59 &plusmn; 6</td>
        </tr>
        <tr>
            <td rowspan="3">BCOH (temporal, single-stream)</td>
            <td>Resnet</td>
            <td>1 &plusmn; 2</td>
            <td>46 &plusmn; 13</td>
            <td>53 &plusmn; 11</td>
        </tr>
        <tr>
            <td>Swin</td>
            <td>1 &plusmn; 2</td>
            <td>36 &plusmn; 10</td>
            <td>63 &plusmn; 10</td>
        </tr>
        <tr>
            <td>X3D</td>
            <td>37 &plusmn; 28</td>
            <td>26 &plusmn; 14</td>
            <td>37 &plusmn; 16</td>
        </tr>
        <tr>
            <td rowspan="3">BCSO (non-temporal)</td>
            <td>Resnet</td>
            <td>5 &plusmn; 6</td>
            <td>23 &plusmn; 7</td>
            <td><strong>72</strong> &plusmn; 4</td>
        </tr>
        <tr>
            <td>Swin</td>
            <td>2 &plusmn; 3</td>
            <td>38 &plusmn; 8</td>
            <td>60 &plusmn; 7</td>
        </tr>
        <tr>
            <td>X3D</td>
            <td>13 &plusmn; 12</td>
            <td>47 &plusmn; 11</td>
            <td>40 &plusmn; 8</td>
        </tr>
    </tbody>
</table>

## Finding Three - Carla NoCrash Benchmark can be solved without temporal information
The single observation method developed in this thesis achieves perfect success scores in the most challenging setting of the Carla NoCrash Benchmark, demonstrating that temporal information is unnecessary for solving this benchmark. This finding challenges the notion that self-driving models inherently benefit from temporal information in closed-loop evaluation. 
<table>
    <caption><strong>Carla NoCrash Benchmark - Town02 (Test) - Dense Traffic (hardest) - 3 Seed Performance</strong></caption>
    <thead>
        <tr>
            <th>Baseline</th>
            <th>Timeouts &#x2193;</th>
            <th>Collisions &#x2193;</th>
            <th>Success &#x2191;</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>BCSO</td>
            <td>0 &plusmn; 0</td>
            <td>2 &plusmn; 4</td>
            <td><strong>98</strong> &plusmn; 4</td>
        </tr>
    </tbody>
</table>

[1] Codevilla, F., Santana, E., Lopez, A., & Gaidon, A. (2019). Exploring the limitations of behavior cloning for autonomous driving
[2] Wen, C., Lin, J., Qian, J., Gao, Y., & Jayaraman, D. (2021). Keyframe-Focused Visual Imitation Learning
[3] Chuang, C.-C., Yang, D., Wen, C., & Gao, Y. (2022). Resolving Copycat Problems in Visual Imitation Learning via Residual Action Prediction

## Visualisations
The following video showcases TimeFuser solving one route of Carla NoCrash Benchmark without any infractions using waypoint output representation and frame-based steering augmentation.
[Waypoints Video](https://www.youtube.com/watch?v=9L257BPMo-M)

This video shows TimeFuser solving one route of Carla NoCrash Benchmark without any infractions using path/target speed prediction as output representation and triangular steering augmentation.
[Path/target speed Video](https://www.youtube.com/watch?v=sO2gYGF9dEE)
