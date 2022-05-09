# Twenty-Newsgroup-Classification

20 newsgroups classification 문제에 대해 0.75 이상의 정확도를 달성하는 model을 만든다.

## Bayes' theorem

Naive Bayes Classifier를 사용할 것이므로, 먼저 베이즈 정리를 이해한다.

베이즈 정리(Bayes’ thorem)는 두 확률변수의 사전확률과 사후확률(조건부확률) 사이의 관계를 나타내는 정리이다.

사전확률로부터 사후확률을 구하는 것은, 어떤 사건이 만들어 놓은 상황에서, 그 사건이 일어난 후 앞으로 일어나게 될 다른 사건의 가능성을 구하는 것을 말한다. 즉, 기존 사건들의 확률(사전확률)을 알고 있다면, 어떤 사건 이후의 조건부 확률을 알 수 있다는 것이다.

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

최적화 문제를 풀 때, 분모의 P(B)는 결과에 영향을 미치지 않으므로 생략할 수도 있다.

## Naive Bayes Classifier

나이브 베이즈 분류기(Naive Bayes Classifier)는 모든 특성들이 독립적이라는 가정을 기반으로 텍스트 문서를 분류하는 대표적인 분류기 중 하나이다.  

텍스트 문서 데이터는 feature matrix와 response(또는 target vector) 두 부분으로 나누어진다.
- feature matrix(X): 각각의 행이 dependent features로 구성된 벡터를 포함하는 행렬이다.
- response/target vector(y): feature matrix의 각 벡터들이 속한 classes를 포함하는 벡터이다.  

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

1. load train data from 20newsgroups dataset and check target_names

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
   
2. fit CountVectorizer with train data and transform train data to CountVectors 

   ```python
   from sklearn.feature_extraction.text import CountVectorizer

   count_vect = CountVectorizer()
   x_train_counts = count_vect.fit_transform(twenty_train.data)
   x_train_counts.shape
   ```
   
3. fit TfidfTransformer with train countVectors and transform train countVectors to TfidfVectors

   ```python
   from sklearn.feature_extraction.text import TfidfTransformer

   tfidf_transformer = TfidfTransformer()
   X_train_tf = tfidf_transformer.fit_transform(x_train_counts)
   X_train_tf.shape 
   ```
   
4. fit MultinomialNB with train TfidfVectors and train target
   
   ```python
   from sklearn.naive_bayes import MultinomialNB

   clf = MultinomialNB().fit(X_train_tf, twenty_train.target)
   ```
   
5. make pipeline and fit model with train data and train target
  
   ```python
   from sklearn.pipeline import Pipeline

   text_clf = Pipeline([ 
       ('vect', CountVectorizer()),
       ('tfidf', TfidfTransformer()),
       ('clf', MultinomialNB())
   ])

   text_clf.fit(twenty_train.data, twenty_train.target)
   ```
   
6. load test data from 20newsgroups dataset and check accuracy

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
   
7. creating confusion matrix and heat map
   
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

