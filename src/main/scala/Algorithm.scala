package org.template.vanilla

import io.prediction.controller.P2LAlgorithm
import io.prediction.controller.Params

import org.apache.commons.math3.random.MersenneTwister
import org.apache.spark.SparkContext

import org.deeplearning4j.models.embeddings.WeightLookupTable
import org.deeplearning4j.models.rntn.RNTN
import org.deeplearning4j.models.word2vec.wordstore.VocabCache
import org.deeplearning4j.models.word2vec.Word2Vec
import org.deeplearning4j.spark.models.word2vec.{Word2Vec => SparkWord2Vec}
import org.deeplearning4j.text.corpora.treeparser.TreeVectorizer

import scala.collection.JavaConversions._

import grizzled.slf4j.Logger

case class AlgorithmParams(mult: Int) extends Params

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
      .setActivationFunction("tanh")
      .setAdagradResetFrequency(1)
      .setCombineClassification(true)
      .setFeatureVectors(word2vec)
      .setRandomFeatureVectors(false)
      .setRng(new MersenneTwister(123))
      .setUseTensors(false)
      .build()
    val listsOfTrees = data.labeledPhrases.mapPartitions(labeledPhrases => {
      val treeVectorizer = new TreeVectorizer() // it is so slow
      labeledPhrases.map(
        x => treeVectorizer.getTreesWithLabels(x.phrase, x.sentiment.toString, data.labels))
    })
    val listOfTrees = listsOfTrees.reduce(_ ++ _)
    rntn.fit(listOfTrees)
    new Model(
      vocabCache = vocabCache,
      weightLookupTable = weightLookupTable,
      rntn = rntn
    )
  }

  def predict(model: Model, query: Query): PredictedResult = {
    val word2Vec = new Word2Vec.Builder()
      .lookupTable(model.weightLookupTable)
      .vocabCache(model.vocabCache)
      .build()
    val nearestWords = word2Vec.wordsNearest(query.q, 10)
    PredictedResult(p = nearestWords.toList)
  }
}

class Model(
  val vocabCache: VocabCache,
  val weightLookupTable: WeightLookupTable,
  val rntn: RNTN
) extends Serializable
