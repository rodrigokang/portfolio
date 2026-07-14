# Sediment Source Modeling

This repository contains the implementation files accompanying the portfolio project **Sediment Source Modeling**, part of the book *Data Science Across Domains: From Methods and Models to Business Applications*.

The project investigates sediment source apportionment using geochemical fingerprinting techniques. Three complementary modelling strategies are implemented and compared:

* **Bootstrapped Mixing Model (BMM)** based on the probabilistic framework proposed by Batista, Laceby, and Evrard (2022).
* **Constrained Least-Squares Mixing Model**, adapted from the deconvolution methodology of McCaffrey for deterministic source proportion estimation.
* **Ratio-Based Constrained Mixing Model**, adapted from the ratio-based deconvolution methodology developed by Sandoval for estimating source contributions using tracer ratios rather than absolute concentrations.

Together, these implementations provide a comparative study of deterministic and probabilistic approaches to sediment source apportionment while illustrating how methodologies originally developed for other geochemical applications can be adapted to environmental systems.

## Repository Structure

* `python/` — Python implementations of the three modelling approaches.
* `data/` — Input datasets required to reproduce the analyses.

Each implementation stores its own figures, tables, and evaluation outputs within its corresponding project directory.

## Dataset

The analyses are based on the experimental sediment fingerprinting dataset published by Batista, Laceby, and Evrard (2022), which contains laboratory and virtual mixtures with known source proportions together with geochemical fingerprint measurements.

The original publication, *A reproducible framework for assessing sediment fingerprinting source apportionment methods*, is available through the Springer website **[here](https://link.springer.com/article/10.1007/s11368-022-03157-4)**.

The input datasets are not included in this repository. Users wishing to reproduce the analyses should obtain the original supplementary material associated with the publication and place the required files in the local `data` directory.

## Companion Chapter

The theoretical background, methodology, model comparisons, results, and discussion are presented in the corresponding chapter of *Data Science Across Domains: From Methods and Models to Business Applications*.

The code contained in this repository provides fully reproducible computational workflows supporting the analyses presented in the chapter.

## Copyright

Copyright © Rodrigo Kang.

All rights reserved.
