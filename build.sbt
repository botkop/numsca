import Dependencies._

lazy val root = (project in file(".")).
  settings(
    inThisBuild(List(
      organization := "be.botkop",
      scalaVersion := "2.12.4",
      version      := "0.1.0-SNAPSHOT"
    )),

    name := "numsca",

    libraryDependencies += "org.nd4j" % "nd4j-native-platform" % "0.9.1",
    libraryDependencies += "com.typesafe.scala-logging" %% "scala-logging" % "3.7.2",
    libraryDependencies += "ch.qos.logback" % "logback-classic" % "1.2.3",

    libraryDependencies += scalaTest % Test
  )

