package org.template.vanilla

import io.prediction.controller.IEngineFactory
import io.prediction.controller.Engine

case class Query(content: String) extends Serializable

case class PredictedResult(
  nearestWords: List[String],
  sentiment: List[Integer]
) extends Serializable

object VanillaEngine extends IEngineFactory {
  def apply() = {
    new Engine(
      classOf[DataSource],
      classOf[Preparator],
      Map("rntn" -> classOf[Algorithm]),
      classOf[Serving])
  }
}
