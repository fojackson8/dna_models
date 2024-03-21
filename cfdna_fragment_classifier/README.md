# Classifying cell-free DNA with multimodal pre-trained model


Large pre-trained model on individual reads from tissue data. Multimodal incorporating sequence + methylation state at base resolution. Apply trained model to cfDNA fragment classification. Steps as follows:

1. Identify DMRs (loaded from another project) 
2. Intersect data at these genomic loci, and convert training data from bam to read format --> `data_process`
3. Train model to predict cancer origin using sequence + methylation state at DMRs, training on tissue data and healthy cfDNA background --> `pre_train`
4. Transfer trained model to predict cancer origin of cfDNA reads. Then summarise predictions to get patient level prediction: cancer | healthy --> `classify_cfdna`


Paper here:
https://openreview.net/forum?id=l7QDJmbxMnL



