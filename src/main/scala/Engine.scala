package org.template.vanilla

import io.prediction.controller.IEngineFactory
import io.prediction.controller.Engine

case class Query(phrase: String) extends Serializable

case class PredictedResult(sentiment: List[Integer]) extends Serializable

object VanillaEngine extends IEngineFactory {
  def apply() = {
    new Engine(
      classOf[DataSource],
      classOf[Preparator],
      Map("rntn" -> classOf[Algorithm]),
      classOf[Serving])
  }
}
