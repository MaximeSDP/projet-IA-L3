## Résultats :

date actualisée : 17/02/2026

DecisionTree : 

Moyenne réussite empirique : 74.33734939759036%
Moyenne réussite réel : 67.94117647058823%
avec congig = === Configuration de l'IA ===
  • path_correct : data/Mer
  • path_incorrect : data/Ailleurs
  • extractors : [<src.extractor.colorHistoExtractor.ColorHistoExtractor object at 0x000001DECCD7BD90>]
  • algo : DecisionTreeClassifier(max_depth=2)
  • train_size : 0.8
  • size_Image : (300, 300)
  • PCA_Active : False


========================================
Moyenne réussite empirique : 73.25301204819277%
Moyenne réussite réelle : 65.91176470588235%
========================================
=== Configuration de l'IA ===
  • path_correct : data/Mer
  • path_incorrect : data/Ailleurs
  • extractors : [colorHistoExtractor, GradientExtractor]
  • algo : DecisionTreeClassifier(max_depth=2)
  • train_size : 0.8
  • size_Image : (300, 300)
  • PCA_Active : False


========================================
Moyenne réussite empirique : 66.74698795180724%
Moyenne réussite réelle : 58.26470588235295%
========================================
=== Configuration de l'IA ===
  • path_correct : data/Mer
  • path_incorrect : data/Ailleurs
  • extractors : [<src.extractor.colorHistoExtractor.ColorHistoExtractor object at 0x00000234DDEAFE50>, <src.extractor.gradientExtractor.GradientExtractor object at 0x00000234C33A4E90>]
  • algo : DecisionTreeClassifier(max_depth=2)
  • train_size : 0.8
  • size_Image : (300, 300)
  • PCA_Active : True


========================================
Moyenne réussite empirique : 70.24096385542167%
Moyenne réussite réelle : 59.74999999999999%
========================================
=== Configuration de l'IA ===
  • path_correct : data/Mer
  • path_incorrect : data/Ailleurs
  • extractors : [<src.extractor.colorHistoExtractor.ColorHistoExtractor object at 0x000001F64E9FFE10>]
  • algo : DecisionTreeClassifier(max_depth=2)
  • train_size : 0.8
  • size_Image : (300, 300)
  • PCA_Active : True


18/02/26 :

SVM :
========================================
Moyenne réussite empirique : 93.954802259887%
Moyenne réussite réelle : 92.34688128772636%
========================================
=== Configuration de l'IA ===
  • path_correct : dataSet/data/train/sea
  • path_incorrect : data/Ailleurs
  • extractors : [<src.extractor.colorHistoExtractor.ColorHistoExtractor object at 0x00000236BFF76510>, <src.extractor.HOGExtractor.HOGExtractor object at 0x00000236F66D8DD0>, <src.extractor.LBPExtractor.LBPExtractor object at 0x00000236F85F6990>, <src.extractor.LBPExtractor.LBPExtractor object at 0x00000236917742D0>]
  • algo : SVC(C=10.0, class_weight='balanced')
  • train_size : 0.8
  • size_Image : (150, 150)
  • PCA_Active : True


  19/02/2026:

  SVM + gridsearch

  ========== GridSearchCV ==========
  Meilleurs paramètres : {'C': 1, 'gamma': 'scale', 'kernel': 'rbf'}
  Meilleur score CV (balanced_accuracy) : 90.83%
==================================


========================================
Erreur empirique (train) : 3.02%
Erreur réelle (CV train) : 9.19%
Accuracy test            : 91.48%
========================================
=== Configuration de l'IA ===
  • path_correct : data/train/positives
  • path_incorrect : data/train/negatives
  • extractors : [<src.extractor.colorHistoExtractor.ColorHistoExtractor object at 0x00000207AAB0A590>, <src.extractor.HSVHistoExtractor.HSVHistoExtractor object at 0x0000020790D2FB90>, <src.extractor.HOGExtractor.HOGExtractor object at 0x00000207AAAFAAD0>, <src.extractor.multiScaleLBPExtractor.MultiScaleLBPExtractor object at 0x00000207AAAFA690>]
  • algo : SVC(class_weight='balanced')
  • train_size : 0.8
  • size_Image : (150, 150)
  • PCA_Active : True
  • grid_search_active : True
  • grid_search_params : {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf'], 'gamma': ['scale']}