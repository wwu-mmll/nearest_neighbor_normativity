
# Nearest Neighbor Normativity N³

This repo contains the code to the paper: [Judged by your neighbors: Brain structural normativity profiles for large and heterogeneous samples](https://www.medrxiv.org/content/10.1101/2024.12.24.24319598v1)


### Abstract
The detection of norm deviations is fundamental to clinical decision making and impacts our ability to diagnose and treat diseases effectively. Current normative modeling approaches rely on generic comparisons and quantify deviations in relation to the population average. However, generic models interpolate subtle nuances and risk the loss of critical information, thereby compromising effective personalization of health care strategies. To acknowledge the substantial heterogeneity  among patients and support the paradigm shift of precision medicine, we introduce Nearest Neighbor Normativity (N³), which is a strategy to refine normativity evaluations in diverse and heterogeneous clinical study populations. We address current methodological shortcomings by accommodating several equally normative population prototypes, comparing individuals from multiple perspectives and designing specifically tailored control groups. Applied to brain structure in 36,896 individuals, the N³ framework provides empirical evidence for its utility and significantly outperforms traditional methods in the detection of pathological alterations. Our results underscore N³’s potential for individual assessments in medical practice, where norm deviations are not merely a benchmark, but an important metric supporting the realization of personalized patient care.

### Code
The core N³ classes are found in the *model* folder. The *KNNProjector* class is the training and application wrapper around the core *KNNOutlier* code, that analyzes the data using the Nearest Neighbor algorithm. 

Statistical analyses are in the *correlation* folder, machine learning analyses in the *ml* folder. 

To train the model, *train_knn.py* should be executed, followed by *apply_knn.py* and *tran_model* in the *normative* folder. 
For subsequent analysis, the AD and MCI data are joined via "data/join_ad.py", before the statistical and machine learning analysis files can be applied. 


