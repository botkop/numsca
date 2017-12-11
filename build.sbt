import Dependencies._

lazy val root = (project in file(".")).settings(
  inThisBuild(
    List(
      organization := "be.botkop",
      scalaVersion := "2.12.4",
      version := "0.1.1-SNAPSHOT"
    )),
  name := "numsca",
  libraryDependencies += "org.nd4j" % "nd4j-native-platform" % "0.9.1",
  libraryDependencies += "com.typesafe.scala-logging" %% "scala-logging" % "3.7.2",
  libraryDependencies += "ch.qos.logback" % "logback-classic" % "1.2.3",
  libraryDependencies += scalaTest % Test
)

publishTo := {
  val nexus = "https://oss.sonatype.org/"
  if (isSnapshot.value)
    Some("snapshots" at nexus + "content/repositories/snapshots")
  else
    Some("releases" at nexus + "service/local/staging/deploy/maven2")
}

pomIncludeRepository := { _ =>
  false
}

licenses := Seq(
  "BSD-style" -> url("http://www.opensource.org/licenses/bsd-license.php"))
homepage := Some(url("https://github.com/botkop"))

scmInfo := Some(
  ScmInfo(
    url("https://github.com/botkop/numsca"),
    "scm:git@github.com:botkop/numsca.git"
  )
)

developers := List(
  Developer(
    id = "botkop",
    name = "Koen Dejonghe",
    email = "koen@botkop.be",
    url = url("https://github.com/botkop")
  )
)

publishMavenStyle := true
publishArtifact in Test := false
skip in publish := true
