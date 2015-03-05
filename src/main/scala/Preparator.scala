package org.template.vanilla

import io.prediction.controller.PPreparator
import io.prediction.data.storage.Event

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.rdd.RDD

class Preparator
  extends PPreparator[TrainingData, PreparedData] {

  def prepare(sc: SparkContext, trainingData: TrainingData): PreparedData = {
    new PreparedData(
      phrases = trainingData.labeledPhrases.map { _.phrase },
      labeledPhrases = trainingData.labeledPhrases
    )
  }
}

class PreparedData(
  val phrases : RDD[String],
  val labeledPhrases: RDD[LabeledPhrase]
) extends Serializable
