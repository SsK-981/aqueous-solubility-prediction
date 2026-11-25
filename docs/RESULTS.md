# Aqueous solubility experiments and findings

This question asks which molecular inputs improve aqueous-solubility prediction and whether those gains carry over to a different dataset. The most informative result emerged on transfer. Plain SchNet estimated absolute solubility more accurately than the custom model, while the custom model better estimated the change between the lowest and highest temperatures in an exploratory profile analysis.

## Experiments on the source datasets

| Experiment | What was tested and what we found |
|---|---|
| **AqSolDBc baseline** | We compared a median predictor, Ridge regression and Random Forest on a fixed scaffold split of 8,047 records. Random Forest had the lowest MAE, 0.885, versus 1.186 for Ridge and 2.050 for the median. The 2,655-record test set contained all 2,023 ring-free records, which matters when interpreting the scores. |
| **BigSolDB fingerprints and temperature** | We grouped 2,336 measurements by supplied compound identity. Across three grouped partitions, adding Morgan fingerprints to the temperature-aware Random Forest reduced mean compound MAE from 0.8582 to 0.7982 in log₁₀ mole fraction. In a separate matched comparison on this cohort, adding temperature lowered observed mean MAE for all eight tested procedures. The cohort includes an unresolved identity conflict and incomplete condition metadata. |
| **SOMAS data audit and chemical features** | Consistent treatment of hydrogen isotopes restored three records. Seventeen records with unresolved CSV/XYZ heavy-element composition disagreements were set aside, leaving 11,679 records. On the subsequent structure-separated test of 876 records, adding functional and supplied quantum descriptors reduced Random Forest MAE from 0.7444 to 0.7392 log₁₀(mol/L). |
| **Internal hydrogen-bond contact feature** | We generated conformers and calculated the fraction that satisfied a distance-and-angle rule. Among the three hydroxybenzaldehyde positional isomers, the rule identified a possible contact for the ortho isomer. On the matched 438-record SOMAS test, adding the feature to the GNN improved one of three seeds. The geometry result establishes a possible internal contact in generated structures. Its effect on measured aqueous solubility remains open. |
| **SchNet and descriptor expansion** | On the same 438 internal SOMAS test records, the custom joint readout lowered the mean of three per-seed MAEs from 0.6902 to 0.6763 versus additive SchNet. In a separate matched MDM comparison, expanding inputs from 10 to 1,219 reduced internal ensemble MAE from 0.8223 to 0.6749. Its compound MAE on transfer changed only from 0.9493 to 0.9488. |

MAE is mean absolute error, and lower values are better. Compound MAE averages the errors for each compound before giving all compounds equal weight. The early temperature-cohort target was log₁₀ mole fraction. SOMAS was scored in log₁₀(mol/L). Each result should be interpreted on its stated split and scale.

## Transfer to BigSolDB

We compared frozen thesis models with an established structure-only baseline, ESOL, on the same 609 release-derived measurements for 81 structures. We also evaluated the 199 measurements for 77 structures between 20 and 30 °C, where ESOL is a closer point-prediction comparator.

| Method | Compound MAE on all records | Compound MAE at 20–30 °C |
|---|---:|---:|
| ESOL with RDKit coefficients | 1.184 | 1.168 |
| Matched Random Forest | 0.895 | 0.895 |
| Plain SchNet | **0.842** | **0.857** |
| Custom SchNet | 0.937 | 0.965 |

Plain SchNet had the lowest compound MAE of the four methods shown here. The larger matched comparison also evaluated GNN and additional controls. ESOL separately scored 0.822 on a prefiltered 56-compound DLS-100 subset. That score is specific to the DLS-100 subset.

## Prediction of the temperature response

We then grouped the saved measurements by structure and source DOI. We retained 82 profiles covering 72 structures with at least three temperatures spanning at least 10 K and a single measurement at each endpoint. We compared the predicted high-minus-low change in log-solubility with the recorded change and weighted structures equally.

| Method | MAE of the predicted change |
|---|---:|
| Assume zero change | 0.659 |
| Matched Random Forest | 0.773 |
| Plain SchNet | 0.593 |
| Custom SchNet | **0.541** |

The custom model had the lowest change error among these four procedures and beat the zero-change control on 50 of 72 structures. This result differs from the absolute-solubility ranking and identifies a narrower question for independent testing. The transfer targets are database-derived dilute concentration estimates. Original tables, solid form and other measurement conditions remain to be checked for the full panel. ESOL training overlap with the transfer compounds remains unaudited. SolProp has yet to be scored on a shared, source-verified panel.

## Reproduction and prior work

See the [test data guide](../data_preparation/README.md) for input preparation and inference commands. The neural model adaptations draw on [Panapitiya et al.](https://doi.org/10.1021/acsomega.2c00642). ESOL uses the fixed [Pat Walters RDKit implementation](https://github.com/PatWalters/solubility), based on [Delaney](https://doi.org/10.1021/ci034243x). [SolProp](https://doi.org/10.1021/jacs.2c01768) is the published temperature-aware comparison for future work on neutral organic solids in water.
