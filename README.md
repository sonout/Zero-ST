# ST-Pattern: Transferring Traffic Predictions to Urban Regions without Target Data

The implementation consists of 5 parts:
1. Feature Extraction: Extract a feature set for representing roads
2. Cluster the traffic on the source city. This is used as labels for the tripplet loss.
3. Road Embedding Learning using tripplet loss: Train the model using the source city and afterwards find the most similar streets for each target street.
4. Create Data for spatio-temporal model: Use the most similar street mapping to create the aproximated historical traffic data for the target city. The nextract daily and weekly periodicity and safe the data for predictior model input.
5. ST-Pred: Given the approximated daily and weekly periodicity, predict the traffic on the target city.

The first three parts are found in the roadembedding folder, while the last two parts are in the st-pred folder.


