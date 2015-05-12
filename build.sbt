import AssemblyKeys._

assemblySettings

name := "template-scala-parallel-dl4j-rntn"

organization := "io.prediction"

resolvers += Resolver.mavenLocal

libraryDependencies ++= Seq(
  "io.prediction"      %% "core"                % pioVersion.value   % "provided",
  "org.apache.spark"   %% "spark-core"          % "1.3.0"   % "provided",
  "org.apache.spark"   %% "spark-mllib"         % "1.3.0"   % "provided",
  "org.deeplearning4j" %  "deeplearning4j-core" % "0.0.3.3.3.alpha1-SNAPSHOT",
  "org.deeplearning4j" %  "deeplearning4j-nlp"  % "0.0.3.3.3.alpha1-SNAPSHOT",
  "org.deeplearning4j" %  "dl4j-spark"          % "0.0.3.3.3.alpha1-SNAPSHOT"
    exclude("org.slf4j", "slf4j-api"),
  "org.deeplearning4j" %  "dl4j-spark-nlp"      % "0.0.3.3.3.alpha1-SNAPSHOT",
  "org.nd4j"           %  "nd4j-jblas"          % "0.0.3.5.5.3-SNAPSHOT",
  "com.google.guava"   %  "guava"               % "14.0.1"
)

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
