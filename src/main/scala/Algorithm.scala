package org.template.vanilla

import io.prediction.controller.P2LAlgorithm
import io.prediction.controller.Params

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.rdd.RDD

import org.deeplearning4j.spark.models.word2vec.{Word2Vec => SparkWord2Vec}
import org.deeplearning4j.models.word2vec.wordstore.VocabCache
import org.deeplearning4j.models.embeddings.WeightLookupTable
import org.deeplearning4j.models.word2vec.Word2Vec
import scala.collection.JavaConversions._

import grizzled.slf4j.Logger

case class AlgorithmParams(mult: Int) extends Params

class Algorithm(val ap: AlgorithmParams)
  extends P2LAlgorithm[PreparedData, Model, Query, PredictedResult] {

  @transient lazy val logger = Logger[this.type]

  def train(sc: SparkContext, data: PreparedData): Model = {
    val trainingResult = new SparkWord2Vec().train(data.phrases)
    new Model(
      vocabCache = trainingResult.getFirst,
      weightLookupTable = trainingResult.getSecond
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
  val weightLookupTable: WeightLookupTable
) extends Serializable {
  override def toString = s"{ vocabCache=[${vocabCache}], weightLookupTable=[${weightLookupTable}]}"
}
