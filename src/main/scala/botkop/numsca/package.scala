package botkop

import org.nd4j.linalg.api.iter.NdIndexIterator
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.api.ops.impl.indexaccum.{IMax, IMin}
import org.nd4j.linalg.api.ops.impl.transforms.custom.{GreaterThanOrEqual, LessThanOrEqual}
import org.nd4j.linalg.api.ops.impl.transforms._
import org.nd4j.linalg.api.ops.random.impl.Choice
import org.nd4j.linalg.api.rng
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms

import scala.collection.JavaConverters._
import scala.language.implicitConversions
import scala.util.Random

package object numsca {

  Nd4j.setDefaultDataTypes(DataType.DOUBLE, DataType.DOUBLE)
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
  def zeros(shape: Long*): Tensor = new Tensor(Nd4j.zeros(shape: _*))
  def zeros(shape: Array[Long]): Tensor = zeros(shape: _*)
  def zerosLike(t: Tensor): Tensor = zeros(t.shape)

  def ones(shape: Long*): Tensor = {
    if (shape.length == 1)
      // probably a bug in nd4j
      new Tensor(Nd4j.ones(1l +: shape: _*))
    else
      new Tensor(Nd4j.ones(shape: _*))
  }
  def ones(shape: Array[Long]): Tensor = ones(shape: _*)

  def full(shape: Array[Long], value: Double): Tensor = zeros(shape) + value

  def randn(shape: Long*): Tensor = new Tensor(Nd4j.randn(shape.toArray: _*))
  def randn(shape: Array[Long]): Tensor = randn(shape: _*)

  def rand(shape: Long*): Tensor = new Tensor(Nd4j.rand(shape.toArray: _*))
  def rand(shape: Array[Long]): Tensor = rand(shape: _*)

  def randint(low: Int, shape: Array[Long]): Tensor = {
    val data = Array.fill(shape.product.toInt)(Random.nextInt(low).toDouble)
    Tensor(data).reshape(shape)
  }
  def randint(low: Int, shape: Long*): Tensor = randint(low, shape.toArray)

  def uniform(low: Double = 0.0,
              high: Double = 1.0,
              shape: Array[Int]): Tensor =
    (new Tensor(Nd4j.randn(shape)) - low) / (high - low)

  def linspace(lower: Double, upper: Double, num: Long): Tensor =
    new Tensor(Nd4j.linspace(lower, upper, num, DataType.DOUBLE))

  def copy(t: Tensor): Tensor = t.copy()

  def abs(t: Tensor): Tensor = new Tensor(Transforms.abs(t.array))

  def maximum(t: Tensor, d: Double): Tensor = t.maximum(d)
  def maximum(a: Tensor, b: Tensor): Tensor = a.maximum(b)
  def minimum(t: Tensor, d: Double): Tensor = t.minimum(d)
  def minimum(a: Tensor, b: Tensor): Tensor = a.minimum(b)

  def max(t: Tensor): Tensor = new Tensor(Nd4j.max(t.array))
  def max(t: Tensor, axis: Int): Tensor = new Tensor(Nd4j.max(t.array, axis))
  def min(t: Tensor): Tensor = new Tensor(Nd4j.min(t.array))
  def min(t: Tensor, axis: Int): Tensor = new Tensor(Nd4j.min(t.array, axis))

  def sum(t: Tensor): Double = Nd4j.sum(t.array).getDouble(0l, 0l)
  def sum(t: Tensor, axis: Int): Tensor = new Tensor(Nd4j.sum(t.array, axis))

  def prod(t: Tensor): Double = Nd4j.prod(t.array).getDouble(0l, 0l)
  def prod(t: Tensor, axis: Int): Tensor = new Tensor(Nd4j.prod(t.array, axis))

  def arange(end: Double): Tensor = new Tensor(Nd4j.arange(end).castTo(DataType.DOUBLE))
  def arange(start: Double, end: Double): Tensor = new Tensor(Nd4j.arange(start, end).castTo(DataType.DOUBLE))

  def sigmoid(t: Tensor): Tensor = new Tensor(Transforms.sigmoid(t.array))
  def softmax(t: Tensor): Tensor = new Tensor(Transforms.softmax(t.array))
  def relu(t: Tensor): Tensor = new Tensor(Transforms.relu(t.array))
  def tanh(t: Tensor): Tensor = new Tensor(Transforms.tanh(t.array))
  def log(t: Tensor): Tensor = new Tensor(Transforms.log(t.array))
  def power(t: Tensor, pow: Double): Tensor =
    new Tensor(Transforms.pow(t.array, pow))
  def exp(t: Tensor): Tensor = new Tensor(Transforms.exp(t.array))
  def sqrt(t: Tensor): Tensor = new Tensor(Transforms.sqrt(t.array))
  def square(t: Tensor): Tensor = power(t, 2)

  def nditer(t: Tensor): Iterator[Array[Long]] = nditer(t.shape)
  def nditer(shape: Array[Long]): Iterator[Array[Long]] =
    new NdIndexIterator(shape: _*).asScala


  def argmax(t: Tensor, axis: Int): Tensor = {
    val indices = Nd4j.getExecutioner.exec(new IMax(t.array, axis)).castTo(DataType.DOUBLE)
    new Tensor(Nd4j.expandDims(indices, indices.rank))
  }
  def argmax(t: Tensor): Tensor = {
    val indices = Nd4j.getExecutioner.exec(new IMax(t.array)).castTo(DataType.DOUBLE)
    new Tensor(Nd4j.expandDims(indices, indices.rank))
  }
  def argmin(t: Tensor, axis: Int): Tensor = {
    val indices = Nd4j.getExecutioner.exec(new IMin(t.array, axis)).castTo(DataType.DOUBLE)
    new Tensor(Nd4j.expandDims(indices, indices.rank))
  }
  def argmin(t: Tensor): Tensor = {
    val indices = Nd4j.getExecutioner.exec(new IMin(t.array)).castTo(DataType.DOUBLE)
    new Tensor(Nd4j.expandDims(indices, indices.rank))
  }

  def round(t: Tensor): Tensor = new Tensor(Transforms.round(t.array))
  def ceil(t: Tensor): Tensor = new Tensor(Transforms.ceil(t.array))
  def floor(t: Tensor): Tensor = new Tensor(Transforms.floor(t.array))

  def mean(t: Tensor): Tensor = new Tensor(Nd4j.mean(t.array))
  def mean(t: Tensor, axis: Int): Tensor = new Tensor(Nd4j.mean(t.array, axis))

  // really do not understand how they calculate the variance and std in nd4j
  def variance(t: Tensor): Tensor = mean((t - mean(t)) ** 2)
  def variance(t: Tensor, axis: Int): Tensor =
    mean((t - mean(t, axis)) ** 2, axis)
  def std(t: Tensor): Tensor = sqrt(variance(t))
  def std(t: Tensor, axis: Int): Tensor = sqrt(variance(t, axis))

  /*
  def variance(t: Tensor): Tensor = new Tensor(Nd4j.`var`(t.array))
  def variance(t: Tensor, axis: Int): Tensor =
    new Tensor(Nd4j.`var`(t.array, axis))
  def std(t: Tensor): Tensor = new Tensor(Nd4j.std(t.array))
  def std(t: Tensor, axis: Int): Tensor = new Tensor(Nd4j.std(t.array, axis))
   */

  def multiply(a: Tensor, b: Tensor): Tensor = a * b
  def dot(a: Tensor, b: Tensor): Tensor = a dot b

  def pad(x: Tensor, padWidth: Array[Array[Int]], mode: Pad.Mode): Tensor = {
    val a = Nd4j.pad(x.array, padWidth, mode, 0.0d)
    new Tensor(a)
  }

  def clip(t: Tensor, min: Double, max: Double): Tensor = t.clip(min, max)

  def concatenate(ts: Seq[Tensor], axis: Int = 0): Tensor =
    new Tensor(Nd4j.concat(axis, ts.map(_.array): _*))

  def reshape(x: Tensor, shape: Array[Long]): Tensor = x.reshape(shape)
  def reshape(x: Tensor, shape: Long*): Tensor = x.reshape(shape: _*)

  def transpose(x: Tensor): Tensor = x.transpose()
  def transpose(x: Tensor, axes: Long*): Tensor = x.transpose(axes: _*)
  def transpose(x: Tensor, axes: Array[Long]): Tensor = x.transpose(axes)

  def arrayEqual(t1: Tensor, t2: Tensor): Boolean = numsca.prod(new Tensor((t1 == t2).array.castTo(DataType.INT), false)) == 1

  def any(x: Tensor): Boolean = {
    require(x.isBoolean)
    sum(new Tensor(x.array.castTo(DataType.INT), false)) > 0
  }

  /*
  def any(x: Tensor, axis: Int): Tensor = {
    throw new NotImplementedError()
  }
   */

  def all(x: Tensor): Boolean = {
    require(x.isBoolean)
    prod(new Tensor(x.array.castTo(DataType.INT), false)) > 0
  }

  /*
  def all(x: Tensor, axis: Int): Tensor = {
    throw new NotImplementedError()
  }
   */

  def choice(a: Tensor, p: Tensor, size: Option[Array[Long]] = None): Tensor = {
    val z = Nd4j.zeros(a.shape: _*)
    Nd4j.getExecutioner.exec(new Choice(a.array, p.array, z))
    if (size.isEmpty) {
      new Tensor(z.getScalar(0l))
    } else {
      new Tensor(z.getScalar(size.get: _*))
    }
  }

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

    /* does not work in nd4j 0.9.1
    def pow(t1: Tensor, t2: Tensor): Tensor = {
      val Seq(ba1, ba2) = tbc(t1, t2)
      // there is a bug in nd4j:
      // c = a ** b
      // updates a, so that a == c
      // therefore we make a copy of a before executing pow
      new Tensor(Transforms.pow(ba1.dup(), ba2))
    }
     */

    def mod(t1: Tensor, t2: Tensor): Tensor = {
      val Seq(ba1, ba2) = tbc(t1, t2)
      new Tensor(ba1.fmod(ba2))
    }

    def gt(t1: Tensor, t2: Tensor): Tensor = {
      val Seq(ba1, ba2) = tbc(t1, t2)
      new Tensor(ba1.gt(ba2).castTo(DataType.DOUBLE), true)
    }

    def gte(t1: Tensor, t2: Tensor): Tensor = {
      new Tensor((gt(t1, t2) || eq(t1, t2)).array.castTo(DataType.DOUBLE), true)
    }

    def lt(t1: Tensor, t2: Tensor): Tensor = {
      val Seq(ba1, ba2) = tbc(t1, t2)
      new Tensor(ba1.lt(ba2).castTo(DataType.DOUBLE), true)
    }

    def lte(t1: Tensor, t2: Tensor): Tensor = {
      new Tensor((lt(t1, t2) || eq(t1, t2)).array.castTo(DataType.DOUBLE), true)
    }

    def eq(t1: Tensor, t2: Tensor): Tensor = {
      val Seq(ba1, ba2) = tbc(t1, t2)
      new Tensor(ba1.eq(ba2).castTo(DataType.DOUBLE), true)
    }

    def neq(t1: Tensor, t2: Tensor): Tensor = {
      val Seq(ba1, ba2) = tbc(t1, t2)
      new Tensor(ba1.neq(ba2).castTo(DataType.DOUBLE), true)
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
        val extShape = Array.fill(diff)(1l)
        a.reshape(extShape ++ a.shape(): _*)
      }
    }

    def broadcastArrays(sa: Seq[INDArray]): Seq[INDArray] = {
      val xa = prepareShapesForBroadcast(sa)
      val rank = xa.head.rank()
      val finalShape: Array[Long] =
        xa.map(_.shape()).foldLeft(Array.fill(rank)(0l)) {
          case (shp, acc) =>
            shp.zip(acc).map { case (a, b) => math.max(a, b) }
        }
      xa.map(a => a.broadcast(finalShape: _*))
    }

    def tbc(ts: Tensor*): Seq[INDArray] = broadcastArrays(ts.map(_.array))

  }

}
