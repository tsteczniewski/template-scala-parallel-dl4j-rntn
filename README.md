# Template description.

This template is based on [deeplearning4j RNTN example](https://github.com/SkymindIO/deeplearning4j-nlp-examples/tree/master/src/main/java/org/deeplearning4j/rottentomatoes/rntn). It's goal is to show how to integrate deeplearning4j library with PredictionIO.

Recursive Neural Tensor Network algorithm is supervised learning algorithm used to predict sentiment of sentences.

As of today, deeplearning4j RNTN algorithm implementation does not work properly (eg. training does not finnish). [Corresponding issue](https://github.com/deeplearning4j/deeplearning4j/issues/225) in deeplearning4j library has been added.

# Installation.

Follow [installation guide for PredictionIO](http://docs.prediction.io/install/).

After installation start all PredictionIO vendors and check pio status:
```bash
pio-start-all
pio status
```

This template depends on deeplearning4j 0.0.3.3.3.alpha1-SNAPSHOT. In order to install it run:
```bash
git clone git@github.com:deeplearning4j/deeplearning4j.git
cd deeplearning4j
chmod a+x setup.sh
./setup.sh
```

Copy this template to your local directory with:
```bash
pio template get ts335793/template-scala-parallel-rntn <TemplateName>
```

# Build, train, deploy.

You might build template, train it and deploy by typing:
```bash
pio build
pio train -- --executor-memory=4GB --driver-memory=4GB
pio deploy -- --executor-memory=4GB --driver-memory=4GB
```
Those pio train options are used to avoid problems with java garbage collector. In case they appear increase executor memory and driver memory.

Attention!
- pio train command won't stop as deeplearning4j RNTN fit function does not work properly

# Importing training data.

You can import example training data from [kaggle](https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews/data). It is collection of the Rotten Tomatoes movie reviews with sentiment labels.

In order to use this data, create new app:
```bash
pio app new <ApplicationName> # prints out ApplicationAccessKey and ApplicationId
```
set appId in engine.json to ApplicationId and import data with:
```bash
python data/import_eventserver.py --access_key <ApplicationAccessKey> --file train.tsv
```

You can always remind your application id and key with:
```bash
pio app list
```

# Sending requests to server.

In order to send a query run in template directory:
```bash
python data/send_query_interactive.py
```
and type phrase you want sentiment to be predicted. The result will be a list of predicted sentiments for all sentences in phrase.

# Algorithm overview.

At first Word2Vec is trained (it creates mapping from words to vectors).
```scala
val (vocabCache, weightLookupTable) = {
  val result = new SparkWord2Vec().train(data.phrases)
  (result.getFirst, result.getSecond)
}
val word2vec = new Word2Vec.Builder()
  .lookupTable(weightLookupTable)
  .vocabCache(vocabCache)
  .build()
```

It is passed to RNTN builder.
```scala
val rntn = new RNTN.Builder()
  .setActivationFunction(ap.activationFunction)
  .setAdagradResetFrequency(ap.adagradResetFrequency)
  .setCombineClassification(ap.combineClassification)
  .setFeatureVectors(word2vec)
  .setRandomFeatureVectors(ap.randomFutureVectors)
  .setRng(new DefaultRandom())
  .setUseTensors(ap.useTensors)
  .build()
```

Each phrase from training set is converted to tree with TreeVectorizer.
```scala
val listsOfTrees = data.labeledPhrases.mapPartitions(labeledPhrases => {
  val treeVectorizer = new TreeVectorizer() // it is so slow
  labeledPhrases.map(
    x => treeVectorizer.getTreesWithLabels(x.phrase, x.sentiment.toString, data.labels))
})
val listOfTrees = listsOfTrees.reduce(_ ++ _)
```

RNTN is fitted to those trees.
```scala
rntn.fit(listOfTrees)
```

Finally, model is saved.
```scala
new Model(
  rntn = rntn,
  labels = data.labels
)
```

# Serving overview.

List of trees for sentences in query is created.
```scala
val trees = new TreeVectorizer().getTreesWithLabels(query.phrase, model.labels)
```

Sentiment for each sentence is being predicted.
```scala
val sentiment = model.rntn.predict(trees)
```

Result is returned.
```scala
PredictedResult(sentiment = sentiment.toList)
```