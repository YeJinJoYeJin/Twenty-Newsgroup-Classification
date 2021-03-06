# Twenty-Newsgroup-Classification

20 newsgroups classification 문제에 대해 0.75 이상의 정확도를 달성하는 model을 만든다.

## Bayes' theorem

Naive Bayes Classifier를 사용할 것이므로, 먼저 베이즈 정리를 이해한다.

베이즈 정리(Bayes’ thorem)는 두 확률변수의 사전확률과 사후확률(조건부확률) 사이의 관계를 나타내는 정리이다.

사전확률로부터 사후확률을 구하는 것은, 어떤 사건이 일어난 후 앞으로 일어나게 될 다른 사건의 가능성을 구하는 것을 말한다. 즉, 기존 사건들의 확률(사전확률)을 알고 있다면 어떤 사건 이후의 조건부 확률을 알 수 있다는 것이다.

베이즈 정리는 확률임에도 귀납적, 경험적인 추론을 사용하며, 불확실성 하에서 의사결정 문제를 수학적으로 다룰 때 중요하게 이용된다.

## 데이터를 이용한 사후확률 추정

어떤 사건이 일어날 확률에 대한 임의의 가정 P(A)에 실제로 발견된 자료나 증거 B를 반영하여, 어떤 사건이 일어날 확률 P(A|B)를 구한다.

![01_베이즈정리](https://user-images.githubusercontent.com/104701375/167359507-a8f6aeae-80db-494c-ac01-2ee2b659ab3f.png)
![02_베이즈정리](https://user-images.githubusercontent.com/104701375/167360130-75bef3a3-cd8a-451f-b099-d149c0b40807.png)

## Naive Bayes Algorithm

사건 B는 여러 개의 사건으로 구성될 수도 있다. 이 경우 베이즈 정리는 다음과 같이 확장된다.

![03_나이브 베이지안 알고리즘](https://user-images.githubusercontent.com/104701375/167360759-00c153b6-edd3-4edf-8d52-6501955c6873.png)

여기서 나이브 베이지안 알고리즘(Naive Bayes Algorithm)은 B를 구성하는 모든 사건들이 서로 독립사건이라 가정한다. 이러한 가정을 통해 우변이 비교적 계산이 쉬운 형태로 변한다.

![04_나이브 베이지안 알고리즘](https://user-images.githubusercontent.com/104701375/167361080-115c71a6-92ef-44a5-9428-6ac1ae49c799.png)

최적화 문제를 풀 때, 분모의 P(B)는 결과에 영향을 미치지 않으므로 생략할 수 있다.

## Naive Bayes Classifier

나이브 베이즈 분류기(Naive Bayes Classifier)는 모든 특성들이 독립적이라는 가정을 기반으로 텍스트 문서를 분류하는 대표적인 분류기 중 하나이다.  

텍스트 문서 데이터는 feature matrix와 response vector(또는 target vector) 두 부분으로 나누어진다.
- feature matrix (X): 각각의 행이 dependent features를 나타내는 벡터를 포함하는 행렬이다.
- response vector/target vector (y): feature matrix의 각 벡터들이 속한 classes를 포함하는 벡터이다.  

베이즈 정리를 통해 X가 주어졌을 때 X가 y에 속할 확률을 다음과 같이 표현할 수 있다.

![05_베이즈정리](https://user-images.githubusercontent.com/104701375/167362726-249e97d0-195e-4d2b-bac8-5d4624236bc4.png)

나이브 베이지안 알고리즘의 형태로 확장하면, 

![06_나이브 베이지안 알고리즘](https://user-images.githubusercontent.com/104701375/167364289-b10467a1-0161-493a-8500-33ea9a2ad397.png)

분모는 결과에 크게 영향을 미치지 않는 상수이므로 생략하면,

![07_나이브 베이지안 알고리즘](https://user-images.githubusercontent.com/104701375/167364626-9c8fa662-3d7a-41e1-ac8e-4a77bfbc83ef.png)

이제 모든 클래스 y에 대해 위와 같은 확률을 구하여, 가장 큰 확률을 갖는 y를 찾는다. 즉, maximum likelihood estimate인 y를 찾아, 해당 문서의 클래스로 분류할 수 있다. 

![08_maximum likelihood estimate](https://user-images.githubusercontent.com/104701375/167365464-7e3461c8-77f4-4a29-8edc-bbe2d258b818.png)

각각의 확률값은 다음과 같이 해석할 수 있다.

![11_확률값 해석](https://user-images.githubusercontent.com/104701375/167369550-2d21598d-fe38-47c5-a152-3f1e6d938e53.png)

이 때 y클래스에 한 번도 등장하지 않은 단어 x가 주어지면 확률값이 0이 되므로, smoothing을 적용하여 다음과 같이 표현할 수 있다. 

![12_확률값 smoothing](https://user-images.githubusercontent.com/104701375/167370335-8eb68322-3718-4482-882b-7b3b6b730685.png)

## Text Classification Using Naive Bayes Classifier

**1. load train data from 20newsgroups dataset and check target_names**

   ```python
   from sklearn.datasets import fetch_20newsgroups

   categories = ['alt.atheism','comp.graphics','comp.os.ms-windows.misc',
    'comp.sys.ibm.pc.hardware','comp.sys.mac.hardware','comp.windows.x',
    'misc.forsale','rec.autos','rec.motorcycles','rec.sport.baseball',
    'rec.sport.hockey','sci.crypt','sci.electronics','sci.med','sci.space',
    'soc.religion.christian','talk.politics.guns','talk.politics.mideast',
    'talk.politics.misc','talk.religion.misc'
   ]

   twenty_train = fetch_20newsgroups(subset='train', categories = categories, shuffle= True, random_state=42)
   twenty_train.target_names
   ```
   
  **2. fit CountVectorizer with train data and transform train data to CountVectors** 

   ```python
   from sklearn.feature_extraction.text import CountVectorizer

   count_vect = CountVectorizer()
   x_train_counts = count_vect.fit_transform(twenty_train.data)
   x_train_counts.shape
   ```
   
   텍스트 문자열의 집합을 token count matrix로 변환시킨다.
   
  **3. fit TfidfTransformer with train countVectors and transform train countVectors to TfidfVectors**

   ```python
   from sklearn.feature_extraction.text import TfidfTransformer

   tfidf_transformer = TfidfTransformer()
   X_train_tf = tfidf_transformer.fit_transform(x_train_counts)
   X_train_tf.shape 
   ```
   count matrix의 값을 정규화된 tf-idf 값으로 변환시킨다. 단어의 빈도를 고려한 적절한 가중치가 각 단어의 값에 곱해진다. 
   
  **4. fit MultinomialNB with train TfidfVectors and train target**
   
   ```python
   from sklearn.naive_bayes import MultinomialNB

   clf = MultinomialNB().fit(X_train_tf, twenty_train.target)
   ```
   
   MultinomialNB 모델은 feature counts 또는 tf-idf fractional counts를 받아 naive bayes algorithm에 따라 maximum likelihood estimate인    category를 찾고, 분류 결과를 출력한다.
   
  **5. make pipeline and fit model with train data and train target**
  
   ```python
   from sklearn.pipeline import Pipeline

   text_clf = Pipeline([ 
       ('vect', CountVectorizer()),
       ('tfidf', TfidfTransformer()),
       ('clf', MultinomialNB())
   ])

   text_clf.fit(twenty_train.data, twenty_train.target)
   ```
   
  **6. load test data from 20newsgroups dataset and check accuracy**

   ```python
   import numpy as np
   from sklearn import metrics

   twenty_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)
   docs_test = twenty_test.data
   predicted = text_clf.predict(docs_test)

   accuracy = np.mean(predicted == twenty_test.target)
   print(f'the accuracy is {accuracy:.3f}')
   bd_accuracy = metrics.balanced_accuracy_score(predicted, twenty_test.target)
   print(f'the balanced accuracy is {bd_accuracy:.3f}')
   ```
   
   ![13_naive bayes classifier accuracy](https://user-images.githubusercontent.com/104701375/167376491-6e501a6b-7a08-49f2-84bd-9174eaff1401.png)
   
  **7. creating confusion matrix and heat map**
   
   ```python
   from sklearn.metrics import confusion_matrix
   import matplotlib.pyplot as plt
   import seaborn as sns

   sns.set(rc = {'figure.figsize':(20,10)})

   mat = confusion_matrix(twenty_test.target, predicted)

   sns.heatmap(mat.T, square=True, annot=True, fmt='d',
   cbar=False, xticklabels = twenty_train.target_names,
   yticklabels= twenty_train.target_names)

   plt.xlabel('true categories')
   plt.ylabel('predicted categories')
   ```
   ![14_naive bayes classifier output](https://user-images.githubusercontent.com/104701375/167376522-23661745-cf27-4ee2-b2bf-328fd72e3251.png)

## Text Classification Using SGDClassifier

SGDClassifier도 MultinomialNB와 마찬가지로 텍스트 문서를 분류할 수 있는 분류기 중 하나이다. 

SGDClassifier는 stochastic gradient descent(SGD) 기법을 적용하여 텍스트 분류 모델을 트레이닝 시키고, SVM 방식의 분류 결과를 출력할 수 있다. 이 때 여러 개의 binary classifiers(one versus all)가 결합되어 multi-class classification이 수행된다. 

![09_SGDClassifier](https://user-images.githubusercontent.com/104701375/167381997-8749a8b9-6379-4317-aa63-3d057510b5d0.png)


정확도를 더 올릴 수 있는지 확인하기 위해 classifier model을 SGDClassifier로 바꿔본다. 

5단계만 다음과 같이 변경되고 나머지 단계는 MultinomialNB를 사용할 때와 동일하다.

   **5. make pipeline and fit model with train data and train target** 
  
   ```python
   from sklearn.pipeline import Pipeline
   from sklearn.linear_model import SGDClassifier

   text_clf = Pipeline([ 
       ('vect', CountVectorizer()),
       ('tfidf', TfidfTransformer()),
       ('clf',SGDClassifier(loss='hinge', penalty='L2',
       alpha=1e-3, random_state=42,
       max_iter=5, tol=None))
   ])

   text_clf.fit(twenty_train.data, twenty_train.target)
   ```
   hinge loss를 설정하여 트레이닝 데이터 각각의 카테고리를 구분하면서 데이터와의 거리가 가장 먼 결정경계를 찾게 된다.
   
   모델의 predictors 수를 줄여서 복잡도를 낮춰주면 overfitting 가능성을 줄이고 정확도를 향상시킬 수 있다. 
   L2 penalty는  모든 predictors의 계수를 제곱한 값의 합이 작아질수록 loss가 작아지게 만든다. 따라서 결과에 영향을 미치는 predictors의 수를 줄여주는 효과를 갖는다. 이 때 alpha는 penalty와 곱해지는 hyperparameter로, alpha 값에 따라 penalty의 강도를 조절할 수 있다.    
   
**결과**

![15_SGDClassifier accuracy](https://user-images.githubusercontent.com/104701375/167381271-14709d6d-df59-40f2-818f-91d25177c6ce.png)

balanced accuracy는 비슷하지만, accuracy가 5% 정도 개선되었다.

![16_SGDClassifier output](https://user-images.githubusercontent.com/104701375/167381279-4020fe39-6bdb-4297-a941-950d8c9d46fb.png)

# 참고자료

- [Bayes’ thorem 1](https://ko.wikipedia.org/wiki/%EB%B2%A0%EC%9D%B4%EC%A6%88_%EC%A0%95%EB%A6%AC)
- [Bayes’ thorem 2](https://namu.wiki/w/%EB%B2%A0%EC%9D%B4%EC%A6%88%20%EC%A0%95%EB%A6%AC)
- [Naive Bayes Algorithm](https://namu.wiki/w/%EB%82%98%EC%9D%B4%EB%B8%8C%20%EB%B2%A0%EC%9D%B4%EC%A7%80%EC%95%88%20%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98)
- [text classification using naïve bayes 1](https://towardsdatascience.com/text-classification-using-naive-bayes-theory-a-working-example-2ef4b7eb7d5a)
- [text classification using naïve bayes 2](https://www.youtube.com/watch?v=xtq-4Q-GK14)
- [stochastic gradient descent 1](https://scikit-learn.org/stable/modules/sgd.html)
- [stochastic gradient descent 2](https://ai92.tistory.com/118)
- [SGDClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html#sklearn.linear_model.SGDClassifier)
