# CS 7180 Spring 2019 Project

This repo contains our group project for [Special Topics in AI: Deep Learning](https://sites.google.com/view/cs-7180-spring-2019/home) taught by Professor [Qi (Rose) Yu](http://roseyu.com/)
in Spring 2019. Our project builds off the [Learning to See in the Dark](http://cchen156.web.engr.illinois.edu/SID.html) paper by Chen et. al. (2018). Their paper introduces an end-to-end
deep learning model which converts a picture taken in the dark to one the is taken in ideal
lighting conditions.

![Chen et. al. 2018](https://github.com/tbonza/CS7180/blob/master/imgs/fig1.png "Results from Chen et. al. 2018")

Our contribution has been to simulate dark conditions on readily available datasets,
rather than using professinal photography equipment and generating relatively few images.
Essentially, we are showing a way forward that achieves similar results to Chen et. al. (2018)
but much more cheaply. Modifying readily available datasets, data augmentation, to achieve
similar or better accuracy allows us to work around the data collection bottleneck currently
experienced by Deep Learning practitioners. We often see a correlation between using more
data and achieving better model accuracy. Given this correlation, we can infer that data 
collection is expensive, therefore, model accuracy is also expensive. We must endeavor to
remove or reduce the data collection bottleneck to improve the adoption of Deep Learning
technologies by industry.

## Project organization

Our project was executed in a proposal, milestone 1, milestone 2, and final report
stage. Our folder structure relates to these stages.

### Proposal

1. Literature review: [lit_review](https://github.com/tbonza/CS7180/tree/master/lit_review)
1. Replication of Chen et. al. (2018): [Learning-to-See-in-the-Dark](https://github.com/cchen156/Learning-to-See-in-the-Dark/tree/f41b54543e908469f84f385da772a4c68538248b)
1. Identifying relevant datasets
   * [ImageNet](https://github.com/tbonza/CS7180/tree/master/ImageNet)
   * [raise dataset](https://github.com/tbonza/CS7180/tree/master/raise%20dataset)
1. Proposal report: [proposal](https://github.com/tbonza/CS7180/tree/master/proposal)
   
### First milestone

1. Replication results and benchmarking: [replication](https://github.com/tbonza/CS7180/tree/master/replication)

### Second milestone

1. Data augmentation of relevant datasets: [simulating_low_light](https://github.com/tbonza/CS7180/tree/master/simulating_low_light)
1. Results generation against relevant datasets: [milestone2](https://github.com/tbonza/CS7180/tree/master/milestone2)

### Final Report

1. [lastmile](https://github.com/tbonza/CS7180/tree/master/lastmile)
	* Results generation for each sub-problem
	* Results from Zero-Shot Learning
	* Results from Transfer Learning
	* Presentation of Results
	
## Hardware specifications

K80

## Framework choice

Tensorflow

## Questions and Improvements


Github issues
