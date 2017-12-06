package botkop

import org.nd4j.linalg.api.iter.NdIndexIterator
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.ops.impl.indexaccum.{IAMax, IAMin}
import org.nd4j.linalg.api.rng
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.factory.Nd4j.PadMode
import org.nd4j.linalg.ops.transforms.Transforms

import scala.collection.JavaConverters._
import scala.language.implicitConversions
import scala.util.Random

package object numsca {

  implicit def selectionToTensor(ts: TensorSelection): Tensor =
    ts.asTensor

  implicit class NumscaDoubleOps(d: Double) {
    def +(t: Tensor): Tensor = t + d
    def -(t: Tensor): Tensor = -t + d
    def *(t: Tensor): Tensor = t * d
    def /(t: Tensor): Tensor = numsca.power(t, -1) * d
  }

  case class NumscaRange(from: Int, to: Option[Int])

  def :>(end: Int) = NumscaRange(0, Some(end))
  def :> = NumscaRange(0, None)

  implicit class NumscaInt(i: Int) {
    def :>(end: Int) = NumscaRange(i, Some(end))
    def :> = NumscaRange(i, None)
  }

  implicit def intToNumscaRange(i: Int): NumscaRange =
    NumscaRange(i, Some(i + 1))

  def rand: rng.Random = Nd4j.getRandom

  def array(ds: Double*) = Tensor(ds: _*)
  def zeros(shape: Int*): Tensor = new Tensor(Nd4j.zeros(shape: _*))
  def zeros(shape: Array[Int]): Tensor = zeros(shape: _*)
  def zerosLike(t: Tensor): Tensor = zeros(t.shape)

  def ones(shape: Int*): Tensor = {
    if (shape.length == 1)
      // probably a bug in nd4j
      new Tensor(Nd4j.ones(1 +: shape: _*))
    else
      new Tensor(Nd4j.ones(shape: _*))
  }
  def ones(shape: Array[Int]): Tensor = ones(shape: _*)

  def randn(shape: Int*): Tensor = new Tensor(Nd4j.randn(shape.toArray))
  def randn(shape: Array[Int]): Tensor = randn(shape: _*)

  def rand(shape: Int*): Tensor = new Tensor(Nd4j.rand(shape.toArray))
  def rand(shape: Array[Int]): Tensor = rand(shape: _*)

  def randint(low: Int, shape: Array[Int]): Tensor = {
    val data = Array.fill(shape.product)(Random.nextInt(low).toDouble)
    Tensor(data).reshape(shape)
  }
  def randint(low: Int, shape: Int*): Tensor = randint(low, shape.toArray)

  def linspace(lower: Double, upper: Double, num: Int): Tensor =
    new Tensor(Nd4j.linspace(lower, upper, num))

  def abs(t: Tensor): Tensor = new Tensor(Transforms.abs(t.array))

  def maximum(t: Tensor, d: Double): Tensor = t.maximum(d)
  def maximum(a: Tensor, b: Tensor): Tensor = a.maximum(b)
  def minimum(t: Tensor, d: Double): Tensor = t.minimum(d)
  def minimum(a: Tensor, b: Tensor): Tensor = a.minimum(b)

  def max(t: Tensor): Tensor = new Tensor(Nd4j.max(t.array))
  def max(t: Tensor, axis: Int): Tensor = new Tensor(Nd4j.max(t.array, axis))
  def min(t: Tensor): Tensor = new Tensor(Nd4j.min(t.array))
  def min(t: Tensor, axis: Int): Tensor = new Tensor(Nd4j.min(t.array, axis))

  def sum(t: Tensor): Double = Nd4j.sum(t.array).getDouble(0, 0)
  def sum(t: Tensor, axis: Int): Tensor = new Tensor(Nd4j.sum(t.array, axis))

  def prod(t: Tensor): Double = Nd4j.prod(t.array).getDouble(0, 0)
  def prod(t: Tensor, axis: Int): Tensor = new Tensor(Nd4j.prod(t.array, axis))

  def arange(end: Double): Tensor = new Tensor(Nd4j.arange(end))
  def arange(start: Double, end: Double): Tensor =
    new Tensor(Nd4j.arange(start, end))

  def sigmoid(t: Tensor): Tensor = new Tensor(Transforms.sigmoid(t.array))
  def relu(t: Tensor): Tensor = new Tensor(Transforms.relu(t.array))
  def tanh(t: Tensor): Tensor = new Tensor(Transforms.tanh(t.array))
  def log(t: Tensor): Tensor = new Tensor(Transforms.log(t.array))
  def power(t: Tensor, pow: Double): Tensor =
    new Tensor(Transforms.pow(t.array, pow))
  def exp(t: Tensor): Tensor = new Tensor(Transforms.exp(t.array))
  def sqrt(t: Tensor): Tensor = new Tensor(Transforms.sqrt(t.array))
  def square(t: Tensor): Tensor = power(t, 2)

  def nditer(t: Tensor): Iterator[Array[Int]] = nditer(t.shape)
  def nditer(shape: Array[Int]): Iterator[Array[Int]] =
    new NdIndexIterator(shape: _*).asScala

  def argmax(t: Tensor): Tensor =
    new Tensor(Nd4j.getExecutioner.exec(new IAMax(t.array)))
  def argmax(t: Tensor, axis: Int): Tensor =
    new Tensor(Nd4j.getExecutioner.exec(new IAMax(t.array), axis))
  def argmin(t: Tensor, axis: Int): Tensor =
    new Tensor(Nd4j.getExecutioner.exec(new IAMin(t.array), axis))
  def argmin(t: Tensor): Tensor =
    new Tensor(Nd4j.getExecutioner.exec(new IAMin(t.array)))

  def round(t: Tensor): Tensor = new Tensor(Transforms.round(t.array))
  def ceil(t: Tensor): Tensor = new Tensor(Transforms.ceil(t.array))
  def floor(t: Tensor): Tensor = new Tensor(Transforms.floor(t.array))

  def mean(t: Tensor): Tensor = new Tensor(Nd4j.mean(t.array))

  def mean(t: Tensor, axis: Int): Tensor = new Tensor(Nd4j.mean(t.array, axis))
  def variance(t: Tensor, axis: Int): Tensor =
    new Tensor(Nd4j.`var`(t.array, axis))

  def multiply(a: Tensor, b: Tensor): Tensor = a * b
  def dot(a: Tensor, b: Tensor): Tensor = a dot b

  def pad(x: Tensor, padWidth: Array[Array[Int]], mode: PadMode): Tensor = {
    val a = Nd4j.pad(x.array, padWidth, mode)
    new Tensor(a)
  }

  def reshape(x: Tensor, shape: Array[Int]): Tensor = x.reshape(shape)
  def reshape(x: Tensor, shape: Int*): Tensor = x.reshape(shape: _*)

  def transpose(x: Tensor): Tensor = x.transpose()
  def transpose(x: Tensor, axes: Int*): Tensor = x.transpose(axes: _*)
  def transpose(x: Tensor, axes: Array[Int]): Tensor = x.transpose(axes)

  def arrayEqual(t1: Tensor, t2: Tensor): Boolean = numsca.prod(t1 == t2) == 1

  def any(x: Tensor): Boolean = {
    require(x.isBoolean)
    sum(x) > 0
  }

  /*
  def any(x: Tensor, axis: Int): Tensor = {
    throw new NotImplementedError()
  }
   */

  def all(x: Tensor): Boolean = {
    require(x.isBoolean)
    prod(x) > 0
  }

  /*
  def all(x: Tensor, axis: Int): Tensor = {
    throw new NotImplementedError()
  }
   */

  // ops between 2 tensors, with broadcasting
  object Ops {

    def add(t1: Tensor, t2: Tensor): Tensor = {
      val Seq(ba1, ba2) = tbc(t1, t2)
      new Tensor(ba1.add(ba2))
    }

    def sub(t1: Tensor, t2: Tensor): Tensor = {
      val Seq(ba1, ba2) = tbc(t1, t2)
      new Tensor(ba1.sub(ba2))
    }

    def mul(t1: Tensor, t2: Tensor): Tensor = {
      val Seq(ba1, ba2) = tbc(t1, t2)
      new Tensor(ba1.mul(ba2))
    }

    def div(t1: Tensor, t2: Tensor): Tensor = {
      val Seq(ba1, ba2) = tbc(t1, t2)
      new Tensor(ba1.div(ba2))
    }

    def mod(t1: Tensor, t2: Tensor): Tensor = {
      val Seq(ba1, ba2) = tbc(t1, t2)
      new Tensor(ba1.fmod(ba2))
    }

    def gt(t1: Tensor, t2: Tensor): Tensor = {
      val Seq(ba1, ba2) = tbc(t1, t2)
      new Tensor(ba1.gt(ba2), true)
    }

    def lt(t1: Tensor, t2: Tensor): Tensor = {
      val Seq(ba1, ba2) = tbc(t1, t2)
      new Tensor(ba1.lt(ba2), true)
    }

    def eq(t1: Tensor, t2: Tensor): Tensor = {
      val Seq(ba1, ba2) = tbc(t1, t2)
      new Tensor(ba1.eq(ba2), true)
    }

    def neq(t1: Tensor, t2: Tensor): Tensor = {
      val Seq(ba1, ba2) = tbc(t1, t2)
      new Tensor(ba1.neq(ba2), true)
    }

    def max(t1: Tensor, t2: Tensor): Tensor = {
      val Seq(ba1, ba2) = tbc(t1, t2)
      new Tensor(Transforms.max(ba1, ba2))
    }

    def min(t1: Tensor, t2: Tensor): Tensor = {
      val Seq(ba1, ba2) = tbc(t1, t2)
      new Tensor(Transforms.min(ba1, ba2))
    }

    def prepareShapesForBroadcast(sa: Seq[INDArray]): Seq[INDArray] = {
      val maxRank = sa.map(_.rank()).max
      sa.map { a =>
        val diff = maxRank - a.rank()
        val extShape = Array.fill(diff)(1)
        a.reshape(extShape ++ a.shape(): _*)
      }
    }

    def broadcastArrays(sa: Seq[INDArray]): Seq[INDArray] = {
      val xa = prepareShapesForBroadcast(sa)
      val rank = xa.head.rank()
      val finalShape: Array[Int] =
        xa.map(_.shape()).foldLeft(Array.fill(rank)(0)) {
          case (shp, acc) =>
            shp.zip(acc).map { case (a, b) => math.max(a, b) }
        }
      xa.map(a => a.broadcast(finalShape: _*))
    }

    def tbc(ts: Tensor*): Seq[INDArray] = broadcastArrays(ts.map(_.array))

  }

}
