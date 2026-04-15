# Accompanying code for "Contrastive Explanations for Anomaly Detection" (TMLR 2025)

This repository contains Python notebooks for reproducing the experimental results of "Contrastive Explanations for Anomaly Detection: Algorithms, Complexity Results and Experiments" (Davidson, 2025).
Experiments are conducted using three datasets: the CelebA celebrity face dataset, the COMPAS recidivism prediction dataset, and the HateXPlain offensive speech dataset.

Deep Support Vector Data Description (DSVDD) is an anomaly detection method that is used in numerous experiments.
Some specialized modifications are added in the following fork: [Deep SVDD Fork](https://github.com/nicbk/Deep-SVDD-PyTorch).

## CelebA
For results involving CelebA, see the `CelebA` directory.
As is specified by filename, group-level explanations are generated using deep support vector data description (DSVDD) anomaly detection and separately with deep convolutional auto-encoder (DCAE) anomaly detection.
For DSVDD experiments, the branch [vae2](https://github.com/nicbk/Deep-SVDD-PyTorch/tree/vae2) from the fork is used.

## COMPAS
For results involving COMPAS, see the `COMPAS` directory.
There, group-level explanations are generated directly with discrete tag data.
There is also an experiment that applies the recidivism scalar score to DSVDD as an alternative basis for applying group-level explanations.
In that case, see the branch [compass](https://github.com/nicbk/Deep-SVDD-PyTorch/tree/compas) from the DSVDD fork.

## HateXPlain
For results involving HateXPlain, see the `HateXPlain` directory.
