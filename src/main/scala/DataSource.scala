package org.template.vanilla

import io.prediction.controller._
import io.prediction.data.storage.Storage

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

import grizzled.slf4j.Logger

case class DataSourceParams(appId: Int) extends Params

class DataSource(val dsp: DataSourceParams)
  extends PDataSource[TrainingData,
      EmptyEvaluationInfo, Query, EmptyActualResult] {

  @transient lazy val logger = Logger[this.type]

  override
  def readTraining(sc: SparkContext): TrainingData = {
    val eventsDb = Storage.getPEvents()
    val eventsRDD: RDD[LabeledPhrase] = eventsDb
      .aggregateProperties(
        appId = dsp.appId,
        entityType = "phrase",
        required = Some(List("sentenceId", "phrase", "sentiment")))(sc)
      .map {
        case (entityId, properties) =>
          LabeledPhrase(
            phraseId = entityId.toInt,
            sentenceId = properties.get[String]("sentenceId").toInt,
            phrase = properties.get[String]("phrase"),
            sentiment = properties.get[String]("sentiment").toInt
          )
      }

    new TrainingData(eventsRDD)
  }
}

case class LabeledPhrase(
  phraseId: Int,
  sentenceId: Int,
  phrase: String,
  sentiment: Int
)

class TrainingData(
  val labeledPhrases: RDD[LabeledPhrase]
) extends Serializable with SanityCheck {
  override def toString = {
    s"events: [${labeledPhrases.count()}] (${labeledPhrases.take(2).toList}...)"
  }

  override def sanityCheck(): Unit = {
    assert(labeledPhrases.count > 0)
  }
}
