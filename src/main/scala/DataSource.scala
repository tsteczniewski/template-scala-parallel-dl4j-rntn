package org.template.vanilla

import io.prediction.controller.PDataSource
import io.prediction.controller.EmptyEvaluationInfo
import io.prediction.controller.EmptyActualResult
import io.prediction.controller.Params
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
    val eventsRDD: RDD[Phrase] = eventsDb
      .aggregateProperties(
        appId = dsp.appId,
        entityType = "phrase",
        required = Some(List("sentenceId", "phrase", "sentiment")))(sc)
      .map {
        case (entityId, properties) =>
          Phrase(
            phraseId = entityId.toInt,
            sentenceId = properties.get[String]("sentenceId").toInt,
            phrase = properties.get[String]("phrase"),
            sentiment = properties.get[String]("sentiment").toInt
          )
      }

    new TrainingData(eventsRDD)
  }
}

case class Phrase(
  phraseId: Int,
  sentenceId: Int,
  phrase: String,
  sentiment: Int
)

class TrainingData(
  val events: RDD[Phrase]
) extends Serializable {
  override def toString = {
    s"events: [${events.count()}] (${events.take(2).toList}...)"
  }
}
