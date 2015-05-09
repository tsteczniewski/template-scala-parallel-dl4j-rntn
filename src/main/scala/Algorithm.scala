package org.template.vanilla

import io.prediction.controller.P2LAlgorithm
import io.prediction.controller.Params

import org.apache.spark.SparkContext

import org.deeplearning4j.models.rntn.RNTN
import org.deeplearning4j.models.word2vec.Word2Vec
import org.deeplearning4j.spark.models.embeddings.word2vec.{Word2Vec => SparkWord2Vec}
import org.deeplearning4j.text.corpora.treeparser.TreeVectorizer
import org.nd4j.linalg.api.rng.DefaultRandom

import scala.collection.JavaConversions._

import grizzled.slf4j.Logger

case class AlgorithmParams(
  activationFunction: String,
  adagradResetFrequency: Integer,
  combineClassification: Boolean,
  randomFutureVectors: Boolean,
  useTensors: Boolean
) extends Params

class Algorithm(val ap: AlgorithmParams)
  extends P2LAlgorithm[PreparedData, Model, Query, PredictedResult] {

  @transient lazy val logger = Logger[this.type]

  def train(sc: SparkContext, data: PreparedData): Model = {
    val (vocabCache, weightLookupTable) = {
      val result = new SparkWord2Vec().train(data.phrases)
      (result.getFirst, result.getSecond)
    }
    val word2vec = new Word2Vec.Builder()
      .lookupTable(weightLookupTable)
      .vocabCache(vocabCache)
      .build()
    val rntn = new RNTN.Builder()
      .setActivationFunction(ap.activationFunction)
      .setAdagradResetFrequency(ap.adagradResetFrequency)
      .setCombineClassification(ap.combineClassification)
      .setFeatureVectors(word2vec)
      .setRandomFeatureVectors(ap.randomFutureVectors)
      .setRng(new DefaultRandom())
      .setUseTensors(ap.useTensors)
      .build()
    val listsOfTrees = data.labeledPhrases.mapPartitions(labeledPhrases => {
      val treeVectorizer = new TreeVectorizer() // it is so slow
      labeledPhrases.map(
        x => treeVectorizer.getTreesWithLabels(x.phrase, x.sentiment.toString, data.labels))
    })
    val listOfTrees = listsOfTrees.reduce(_ ++ _)
    rntn.fit(listOfTrees)
    new Model(
      rntn = rntn,
      labels = data.labels
    )
  }

  def predict(model: Model, query: Query): PredictedResult = {
    val trees = new TreeVectorizer().getTreesWithLabels(query.phrase, model.labels)
    val sentiment = model.rntn.predict(trees)
    PredictedResult(sentiment = sentiment.toList)
  }
}

class Model(
  val rntn: RNTN,
  val labels: List[String]
) extends Serializable
