name := "spark-glmnet"

version := "1.5"

scalaVersion := "2.10.4"

javacOptions ++= Seq("-source", "1.7", "-target", "1.7")

// Must run the examples and tests in separate JVMs to avoid mysterious
// scala.reflect.internal.MissingRequirementError errors.
// https://github.com/deanwampler/spark-workshop/blob/master/project/Build.scala
fork in run := true

// additional libraries
libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "1.4.1" % "provided",
  "org.apache.spark" %% "spark-sql" % "1.4.1" % "provided",
//  "org.apache.spark" %% "spark-mllib" % "1.4.1" % "provided",
  "com.github.fommil.netlib" % "all" % "1.1.2" pomOnly(),
  "org.scalanlp" %% "breeze" % "0.11.2",
  "org.apache.commons" % "commons-lang3" % "3.0",
  "org.jblas" % "jblas" % "1.2.4",
  "com.github.scopt" %% "scopt" % "3.2.0",
  "org.scalatest" %% "scalatest" % "2.2.1" % "test"
)

resolvers ++= Seq(
  "Akka Repository" at "http://repo.akka.io/releases/",
  "Twitter4J Repository" at "http://twitter4j.org/maven2/",
  "Twitter Maven Repo" at "http://maven.twttr.com/",
  "scala-tools" at "https://oss.sonatype.org/content/groups/scala-tools",
  "Typesafe repository" at "http://repo.typesafe.com/typesafe/releases/",
  "Second Typesafe repo" at "http://repo.typesafe.com/typesafe/maven-releases/",
  "Mesosphere Public Repository" at "http://downloads.mesosphere.io/maven",
  Resolver.sonatypeRepo("public")
)

assemblyOption in assembly := (assemblyOption in assembly).value.copy(includeScala = false, includeDependency = true)

assemblyMergeStrategy in assembly := {
  case m if m.toLowerCase.endsWith("manifest.mf") => MergeStrategy.discard
  case m if m.startsWith("META-INF") => MergeStrategy.discard
  case PathList("javax", "servlet", xs @ _*) => MergeStrategy.first
  case PathList("org", "apache", xs @ _*) => MergeStrategy.first
  case PathList("org", "jboss", xs @ _*) => MergeStrategy.first
  case "about.html" => MergeStrategy.rename
  case "reference.conf" => MergeStrategy.concat
  case _ => MergeStrategy.first
}
