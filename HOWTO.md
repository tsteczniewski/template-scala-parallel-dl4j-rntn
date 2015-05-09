# Integrating Deeplearnign4j library

This file presents steps that must be taken in order to change PredictionIO vanilla template to this template and integrate Deeplearning4j library.

## Install Deeplearning4j 0.0.3.3.3.alpha1-SNAPSHOT

```bash
git clone git@github.com:deeplearning4j/deeplearning4j.git
cd deeplearning4j
chmod a+x setup.sh
./setup.sh
```

## Modify build.sbt

In order to use Deeplearning4j in your template you must add it along with libraries it depends on to dependencies in your build.sbt.
```scala
libraryDependencies ++= Seq(
  "io.prediction"      %% "core"                % "0.9.2"   % "provided",
  "org.apache.spark"   %% "spark-core"          % "1.3.0"   % "provided",
  "org.apache.spark"   %% "spark-mllib"         % "1.3.0"   % "provided",
  "org.deeplearning4j" %  "deeplearning4j-core" % "0.0.3.3.3.alpha1-SNAPSHOT", // ADDED
  "org.deeplearning4j" %  "deeplearning4j-nlp"  % "0.0.3.3.3.alpha1-SNAPSHOT", // ADDED
  "org.deeplearning4j" %  "dl4j-spark"          % "0.0.3.3.3.alpha1-SNAPSHOT" 
    exclude("org.slf4j", "slf4j-api"), // ADDED
  "org.deeplearning4j" %  "dl4j-spark-nlp"      % "0.0.3.3.3.alpha1-SNAPSHOT", // ADDED
  "org.nd4j"           %  "nd4j-jblas"          % "0.0.3.5.5.3-SNAPSHOT", //ADDED
  "com.google.guava"   %  "guava"               % "14.0.1" // ADDED
)
```

In order to handle dependencies conflicts when installing deeplearning4j add this merge strategy to build.sbt.
```scala
// ADDED
mergeStrategy in assembly := {
  case x if Assembly.isConfigFile(x) =>
    MergeStrategy.concat
  case PathList(ps @ _*) if Assembly.isReadme(ps.last) || Assembly.isLicenseFile(ps.last) =>
    MergeStrategy.rename
  case PathList("META-INF", xs @ _*) =>
    (xs map {_.toLowerCase}) match {
      case ("manifest.mf" :: Nil) | ("index.list" :: Nil) | ("dependencies" :: Nil) =>
        MergeStrategy.discard
      case ps@(x :: xs) if ps.last.endsWith(".sf") || ps.last.endsWith(".dsa") =>
        MergeStrategy.discard
      case "plexus" :: xs =>
        MergeStrategy.discard
      case "services" :: xs =>
        MergeStrategy.filterDistinctLines
      case ("spring.schemas" :: Nil) | ("spring.handlers" :: Nil) =>
        MergeStrategy.filterDistinctLines
      case _ => MergeStrategy.first
    }
  case PathList(_*) => MergeStrategy.first
}
```

## Update Engine.scala

Update Query to include phrase, and the Predict Result to include list of predicted sentiments for all sentences in query.
```scala
case class Query(phrase: String) extends Serializable // CHANGED

case class PredictedResult(sentiment: List[Integer]) extends Serializable // CHANGED
```

Change algorithm name.
```scala
object VanillaEngine extends IEngineFactory {
  def apply() = {
    new Engine(
      classOf[DataSource],
      classOf[Preparator],
      Map("rntn" -> classOf[Algorithm]), // CHANGED
      classOf[Serving])
  }
}
```

## Update engine.json

Deeplearning4j spark Word2Vec requires setting "negative" option to zero in spark config. Set algorithm parameters.
```scala
{
  "id": "default",
  "description": "Default settings",
  "engineFactory": "org.template.vanilla.VanillaEngine",
  "sparkConf": {
    // ADDED
    "org.deeplearning4j.scaleout.perform.models.word2vec": {
      "negative": 0
    }
  },
  "datasource": {
    "params" : {
      "appId": 1
    }
  },
  "algorithms": [
    {
      "name": "rntn", // CHANGED
      // ADDED
      "params": {
        "activationFunction": "tanh",
        "adagradResetFrequency": 1,
        "combineClassification": true,
        "randomFutureVectors": false,
        "useTensors": false
      }
    }
  ]
}
```

## Update import_eventserver.py

Adjust data/import_eventserver.py to Kaggle's data.

## Update DataSource.scala

Create Labeled Phrase class representing one phrase from data set and modify Training Data to return list of Labeled Phrases. Add sanity check that will be runned each time training will be executed.
```scala
// ADDED
case class LabeledPhrase(
  phraseId: Int,
  sentenceId: Int,
  phrase: String,
  sentiment: Int
)

class TrainingData(
  val labeledPhrases: RDD[LabeledPhrase] // CHANGED
) extends Serializable with SanityCheck { // CHANGED
  // CHANGED
  override def toString = {
    s"events: [${labeledPhrases.count()}] (${labeledPhrases.take(2).toList}...)"
  }

  // ADDED
  override def sanityCheck(): Unit = {
    assert(labeledPhrases.count > 0)
  }
}
```

Adjust readTrainging to load Kaggle's data from database.
```scala
class DataSource(val dsp: DataSourceParams)
  extends PDataSource[TrainingData,
      EmptyEvaluationInfo, Query, EmptyActualResult] {

  @transient lazy val logger = Logger[this.type]
  
  // CHANGED
  override
  def readTraining(sc: SparkContext): TrainingData = {
    val eventsDb = Storage.getPEvents()
    val eventsRDD: RDD[LabeledPhrase] = eventsDb
      .aggregateProperties(
        appId = dsp.appId,
        entityType = "phrase",
        required = Some(List("sentenceId", "phrase", "sentiment")))(sc)
      .map({
        case (entityId, properties) =>
          LabeledPhrase(
            phraseId = entityId.toInt,
            sentenceId = properties.get[String]("sentenceId").toInt,
            phrase = properties.get[String]("phrase"),
            sentiment = properties.get[String]("sentiment").toInt
          )
      })

    new TrainingData(eventsRDD)
  }
}
```

## Update Preparator.scala

Make Prepared Data for store
* phrases strings
* labeled phrases
* sentiment classes (numbers from 0 to 4 in case of Kaggle's data)
```scala
class PreparedData(
  val phrases : RDD[String],
  val labeledPhrases: RDD[LabeledPhrase],
  val labels : List[String]
) extends Serializable
```

Convert data from Data Source to Prepared Data.
```scala
class Preparator
  extends PPreparator[TrainingData, PreparedData] {

  // ADDED
  def prepare(sc: SparkContext, trainingData: TrainingData): PreparedData = {
    new PreparedData(
      phrases = trainingData.labeledPhrases.map { _.phrase },
      labeledPhrases = trainingData.labeledPhrases,
      labels = "0" :: "1" :: "2" :: "3" :: "4" :: Nil
    )
  }
}
```

### Modify Algorithm.scala

Include Deeplearning4j libraries.
```scala
import org.deeplearning4j.models.rntn.RNTN // ADDED
import org.deeplearning4j.models.word2vec.Word2Vec // ADDED
import org.deeplearning4j.spark.models.embeddings.word2vec.{Word2Vec => SparkWord2Vec} // ADDED
import org.deeplearning4j.text.corpora.treeparser.TreeVectorizer // ADDED
import org.nd4j.linalg.api.rng.DefaultRandom // ADDED
```

Change AlgorithmParams to store information about number of nearest words returned in query result.
```scala
// CHANGED
case class AlgorithmParams(
  activationFunction: String,
  adagradResetFrequency: Integer,
  combineClassification: Boolean,
  randomFutureVectors: Boolean,
  useTensors: Boolean
) extends Params
```

Make Model store RNTN model.
```scala
// CHANGED
class Model(
  val rntn: RNTN,
  val labels: List[String]
) extends Serializable
```
### Modify train function

At first train Word2Vec (it creates mapping from words to vectors).
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

Create RNTN with vocabCache, weightLookupTable and AlgorithmParams.
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

Convert each phrase from training set to tree with TreeVectorizer.
```scala
val listsOfTrees = data.labeledPhrases.mapPartitions(labeledPhrases => {
  val treeVectorizer = new TreeVectorizer() // it is so slow
  labeledPhrases.map(
    x => treeVectorizer.getTreesWithLabels(x.phrase, x.sentiment.toString, data.labels))
})
val listOfTrees = listsOfTrees.reduce(_ ++ _)
```

Fit RNTN with those trees.
```scala
rntn.fit(listOfTrees)
```

Save model.
```scala
new Model(
  rntn = rntn,
  labels = data.labels
)
```

### Modify predict function

Do:
* convert query phrase to tree
* use RNTN to predict sentiment
* return list of predicted results
```scala
def predict(model: Model, query: Query): PredictedResult = {
  val trees = new TreeVectorizer().getTreesWithLabels(query.phrase, model.labels)
  val sentiment = model.rntn.predict(trees)
  PredictedResult(sentiment = sentiment.toList)
}
```