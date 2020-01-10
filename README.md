# Readmissions
A hospital readmission is an episode when a patient who had been discharged from a hospital is admitted again within a specified time interval. Readmission rates have increasingly been used as an outcome measure in health services research and as a quality benchmark for health systems. Hospital readmission rates were formally included in reimbursement decisions for the Centers for Medicare and Medicaid Services (CMS) as part of the Patient Protection and Affordable Care Act (ACA) of 2010, which penalizes health systems with higher than expected readmission rates through the Hospital Readmission Reduction Program.  The data in this problem is real-world data covering 10 years (1999–2008) of clinical care at 130 hospitals and integrated delivery networks throughout the United States. 

The specific patient cases used in this challenge also meet the following criteria:
it is generated from a hospital admission with length of stay between 1 and 14 days
diabetes was entered to the system as one of the diagnoses
laboratory tests were performed during the encounter
medications were administered during the encounter

In this machine learning problem, you have the exacting task of trying to predict whether or not a patient will be readmitted to the hospital. The target here takes on binary values where 0 implies that the patient was not readmitted, and 1 implies that the patient was readmitted. You are predicting the latter.

Your classifcation model will be evaluated using Log Loss. Your predictions, therefore, will be probabilities of readmission (see the "Evaluation" section for more details).

In the data section, you can download hm7-train.csv to help you develop your model. Additionally, a hm7-test data file is available which does not include the readmitted variable.

Approximately 30% of the hm7-test data is used to calculate a public log loss value -- you and all other competing teams can see this value. The remaining 70% of the test data is used to evaluate your private competition score -- no one other than the administrators can see this score until the competition is closed. The final quantitative ranking of your model performance will be based on this 70% holdout data set. You can upload new predictions to Kaggle to receive feedback on your model performance up to 3 times per day.
