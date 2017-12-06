package botkop.numsca

import botkop.{numsca => ns}
import org.scalatest.{FlatSpec, Matchers}

import scala.language.postfixOps

class NumscaSpec extends FlatSpec with Matchers {

  val ta: Tensor = ns.arange(10)
  val tb: Tensor = ns.reshape(ns.arange(9), 3, 3)
  val tc: Tensor = ns.reshape(ns.arange(2 * 3 * 4), 2, 3, 4)

  "A Tensor" should "transpose over multiple dimensions" in {
    val x = ns.arange(6).reshape(1, 2, 3)
    val y = ns.transpose(x, 1, 0, 2)
    val z = ns.reshape(x, 2, 1, 3)
    assert(ns.arrayEqual(y, z))
  }

  // tests based on http://scipy-cookbook.readthedocs.io/items/Indexing.html

  // Elements
  it should "retrieve the correct elements" in {
    // todo: implicitly convert tensor to double when only 1 element?
    assert(ta(1).squeeze() == 1)
    assert(tb(1, 0).squeeze() == 3)
    assert(tc(1, 0, 2).squeeze() == 14)

    val i = List(1, 0, 1)
    assert(tc(i: _*).squeeze() == 13)

  }

  it should "access through an iterator" in {

    val expected = List(
      (Array(0, 0), 0.00),
      (Array(0, 1), 1.00),
      (Array(0, 2), 2.00),
      (Array(1, 0), 3.00),
      (Array(1, 1), 4.00),
      (Array(1, 2), 5.00),
      (Array(2, 0), 6.00),
      (Array(2, 1), 7.00),
      (Array(2, 2), 8.00)
    )

    ns.nditer(tb.shape).zipWithIndex.foreach {
      case (i1, i2) =>
        assert(i1 sameElements expected(i2)._1)
        assert(tb(i1).squeeze() == expected(i2)._2)
    }
  }

  it should "change array values in place" in {
    val t = ta.copy()
    t(3) := -5
    assert(t.data sameElements Array(0, 1, 2, -5, 4, 5, 6, 7, 8, 9))
    t(0) += 7
    assert(t.data sameElements Array(7, 1, 2, -5, 4, 5, 6, 7, 8, 9))

    val t2 = tb.copy()
    val x = Array(2, 1)
    t2(x) := -7
    t2(1, 2) := -3
    assert(
      arrayEqual(t2,
                 Tensor(0.00, 1.00, 2.00, 3.00, 4.00, -3.00, 6.00, -7.00,
                   8.00).reshape(3, 3)))
  }

  it should "do operations array-wise" in {
    val a2 = 2 * ta
    assert(a2.data sameElements Array(0, 2, 4, 6, 8, 10, 12, 14, 16, 18))
  }

  it should "slice over a single dimension" in {
    println(ta.shape.toList)

    // turn into a column vector
    val a0 = ta.copy().reshape(10, 1)

    // A[1:]
    val a1 = a0(1 :>)

    // A[:-1]
    val a2 = a0(0 :> -1)

    // A[1:] - A[:-1]
    val a3 = a1 - a2

    assert(ns.arrayEqual(a3, Tensor(1, 1, 1, 1, 1, 1, 1, 1, 1)))

    assert(ns.arrayEqual(ta(:>, 5 :>), Tensor(5, 6, 7, 8, 9)))
    assert(ns.arrayEqual(ta(:>, :>(5)), Tensor(0, 1, 2, 3, 4)))
    assert(ns.arrayEqual(ta(:>, -3 :>), Tensor(7, 8, 9)))
  }

  it should "update over a single dimension" in {
    val t = ta.copy()
    t(2 :> 5) := -ns.ones(3)
    val e1 =
      Tensor(0.00, 1.00, -1.00, -1.00, -1.00, 5.00, 6.00, 7.00, 8.00, 9.00)
    assert(ns.arrayEqual(t, e1))

    an[IllegalStateException] should be thrownBy {
      t(2 :> 5) := -ns.ones(4)
    }

    // this does not throw an exception !!!
    /*
    an[IllegalStateException] should be thrownBy {
      t(2 :> 6) := -ns.ones(4).reshape(2, 2)
    }
     */

    t(2 :> 5) := 33
    assert(
      ns.arrayEqual(
        t,
        Tensor(0.00, 1.00, 33.00, 33.00, 33.00, 5.00, 6.00, 7.00, 8.00, 9.00)))

    t(2 :> 5) -= 1
    assert(
      ns.arrayEqual(
        t,
        Tensor(0.00, 1.00, 32.00, 32.00, 32.00, 5.00, 6.00, 7.00, 8.00, 9.00)))

    t := -1
    assert(
      ns.arrayEqual(t,
                    Tensor(-1.00, -1.00, -1.00, -1.00, -1.00, -1.00, -1.00,
                      -1.00, -1.00, -1.00)))

    val s = 3 :> -1
    assert(ns.arrayEqual(ta(:>, s), Tensor(3.00, 4.00, 5.00, 6.00, 7.00, 8.00)))

  }

  it should "slice over multiple dimensions" in {
    val b1 = tb(0 :> 2, :>)
    assert(ns.arrayEqual(b1, Tensor(0, 1, 2, 3, 4, 5).reshape(2, 3)))
  }

  it should "slice over multiple dimensions with integer indexing" in {
    val b2 = tb(1, 0 :> -1)
    assert(ns.arrayEqual(b2, Tensor(3.00, 4.00)))
  }

  it should "broadcast with another tensor" in {

    // tests inspired by
    // https://docs.scipy.org/doc/numpy-1.13.0/user/basics.broadcasting.html
    // http://scipy.github.io/old-wiki/pages/EricsBroadcastingDoc

    def verify(shape1: Array[Int],
               shape2: Array[Int],
               expectedShape: Array[Int]) = {
      val t1 = ns.ones(shape1)
      val t2 = ns.ones(shape2)
      val Seq(s1, s2) = Ops.tbc(t1, t2)
      assert(s1.shape().sameElements(s2.shape()))
      assert(s1.shape().sameElements(expectedShape))
    }

    verify(Array(8, 1, 6, 1), Array(7, 1, 5), Array(8, 7, 6, 5))
    verify(Array(256, 256, 3), Array(3), Array(256, 256, 3))
    verify(Array(5, 4), Array(1), Array(5, 4))
    verify(Array(15, 3, 5), Array(15, 1, 5), Array(15, 3, 5))
    verify(Array(15, 3, 5), Array(3, 5), Array(15, 3, 5))
    verify(Array(15, 3, 5), Array(3, 1), Array(15, 3, 5))

    val x = ns.arange(4)
    val xx = x.reshape(4, 1)
    val y = ns.ones(5)
    val z = ns.ones(3, 4)

    an[IllegalArgumentException] should be thrownBy x + y

    (xx + y).shape shouldBe Array(4, 5)
    val s1 =
      Tensor(
        1, 1, 1, 1, 1, //
        2, 2, 2, 2, 2, //
        3, 3, 3, 3, 3, //
        4, 4, 4, 4, 4 //
      ).reshape(4, 5)
    assert(ns.arrayEqual(xx + y, s1))

    (x + z).shape shouldBe Array(3, 4)
    val s2 =
      Tensor(
        1, 2, 3, 4, //
        1, 2, 3, 4, //
        1, 2, 3, 4 //
      ).reshape(3, 4)
    assert(ns.arrayEqual(x + z, s2))

    // outer sum
    val a = Tensor(0.0, 10.0, 20.0, 30.0).reshape(4, 1)
    val b = Tensor(1.0, 2.0, 3.0)
    val c = Tensor(
      1.00, 2.00, 3.00, //
      11.00, 12.00, 13.00, //
      21.00, 22.00, 23.00, //
      31.00, 32.00, 33.00 //
    ).reshape(4, 3)

    assert(ns.arrayEqual(a + b, c))

    // vector quantization
    val observation = Tensor(111.0, 188.0)
    val codes = Tensor(
      102.0, 203.0, //
      132.0, 193.0, //
      45.0, 155.0, //
      57.0, 173.0 //
    ).reshape(4, 2)
    val diff = codes - observation
    val dist = ns.sqrt(ns.sum(ns.square(diff), axis = -1))
    val nearest = ns.argmin(dist).squeeze()
    assert(nearest == 0.0)

    // also test same shape
    val t1 = ns.arange(9).reshape(3, 3)
    val t2 = t1 + t1
    assert(ns.arrayEqual(t2, t1 * 2))
  }

  it should "do boolean indexing" in {
    val c = ta < 5 && ta > 1
    // c = [0.00,  0.00,  1.00,  1.00,  1.00,  0.00,  0.00,  0.00,  0.00,  0.00]
    val d = ta(c)
    assert(ns.arrayEqual(d, Tensor(2, 3, 4)))
  }

  it should "update with boolean indexing along a single dimension" in {
    val c = ta < 5 && ta > 1
    val t = ta.copy()
    t(c) := -7
    assert(
      ns.arrayEqual(
        t,
        Tensor(0.00, 1.00, -7.00, -7.00, -7.00, 5.00, 6.00, 7.00, 8.00, 9.00)))
  }

  it should "do multidimensional boolean indexing" in {
    val c = tc(tc % 5 == 0)
    assert(ns.arrayEqual(c, Tensor(0.00, 5.00, 10.00, 15.00, 20.00)))
    assert(ns.any(tb < 5))
    assert(!ns.all(tb < 5))

    // stuff like this is not yet implemented
    // np.any(B<5, axis=1)
    // B[np.any(B<5, axis=1),:]
    // c = np.any(C<5,axis=2)
  }

  it should "update with boolean indexing along multiple dimensions" in {
    val c = tb < 5 && tb > 1
    val t1 = tb.copy()
    t1(c) := -7

    assert(
      ns.arrayEqual(t1, Tensor(0, 1, -7, -7, -7, 5, 6, 7, 8).reshape(3, 3)))

    val t2 = tb.copy()
    t2(c) += 10
    println(t2)
    assert(
      ns.arrayEqual(t2, Tensor(0, 1, 12, 13, 14, 5, 6, 7, 8).reshape(3, 3)))

    // val t3 = tb.copy()
    // t3.put((ix, d) => { d - t3(ix).squeeze() })(c)
    // assert(ns.arrayEqual(t3, Tensor(0, 1, 0, 0, 0, 5, 6, 7, 8).reshape(3, 3)))
  }

  it should "do list-of-location indexing" in {
    val primes = Tensor(2, 3, 5, 7, 11, 13, 17, 19, 23)
    val idx = Tensor(3, 4, 1, 2, 2)
    val r = primes(idx)
    assert(ns.arrayEqual(r, Tensor(7.00, 11.00, 3.00, 5.00, 5.00)))

    val r2 = primes(tb)
    val e2 = Tensor(
      2.00, 3.00, 5.00, //
      7.00, 11.00, 13.00, //
      17.00, 19.00, 23.00 //
    ).reshape(3, 3)
    assert(ns.arrayEqual(r2, e2))

    val tp = primes.reshape(3, 3)
    a[NotImplementedError] should be thrownBy tp(tb)

    val numSamples = 4
    val numClasses = 3
    val x = ns.arange(numSamples * numClasses).reshape(numSamples, numClasses)
    val y = Tensor(0, 1, 2, 1)
    val z = x(ns.arange(numSamples), y)
    assert(ns.arrayEqual(z, Tensor(0.00, 4.00, 8.00, 10.00)))

  }

  it should "update along a single dimension" in {
    val primes = Tensor(2, 3, 5, 7, 11, 13, 17, 19, 23)
    val idx = Tensor(3, 4, 1, 2, 2)
    primes(idx) := 0
    assert(
      ns.arrayEqual(
        primes,
        Tensor(2.00, 0.00, 0.00, 0.00, 0.00, 13.00, 17.00, 19.00, 23.00)))
  }

  it should "do multi dim list-of-location indexing" in {
    val a = ns.arange(6).reshape(3, 2) + 1
    val s1 = Tensor(0, 1, 2)
    val s2 = Tensor(0, 1, 0)
    val r1 = a(s1, s2)
    assert(ns.arrayEqual(r1, Tensor(1, 4, 5)))

    val s3 = Tensor(0, 0)
    val s4 = Tensor(1, 1)
    val r2 = a(s3, s4)
    assert(ns.arrayEqual(r2, Tensor(2, 2)))

    val b = ns.arange(12).reshape(4, 3) + 1
    val c = Tensor(0, 2, 0, 1)
    val r3 = b(arange(4), c)
    assert(ns.arrayEqual(r3, Tensor(1, 6, 7, 11)))

    val y = ns.arange(35).reshape(5, 7)
    val r4 = y(Tensor(0, 2, 4), Tensor(0, 1, 2))
    assert(ns.arrayEqual(r4, Tensor(0, 15, 30)))

    // broadcast selection
    val r5 = y(Tensor(0, 2, 4), Tensor(1))
    assert(ns.arrayEqual(r5, Tensor(1, 15, 29)))
  }

  it should "update along multiple dimensions" in {
    val a = ns.arange(6).reshape(3, 2) + 1
    val s1 = Tensor(1, 1, 2)
    val s2 = Tensor(0, 1, 0)

    a(s1, s2) := 0
    assert(ns.arrayEqual(a, Tensor(1, 2, 0, 0, 0, 6).reshape(3, 2)))

    a(s1, s2) += 3000
    assert(ns.arrayEqual(a, Tensor(1, 2, 3000, 3000, 3000, 6).reshape(3, 2)))
  }

  it should "do other calculations" in {
    val a = tb.copy()
    val s1 = Tensor(1, 1, 2)
    val s2 = Tensor(0, 1, 0)

    val c = a(s1, s2) + 3
    assert(ns.arrayEqual(c, Tensor(6, 7, 9)))
  }

  it should "mimic the smallest neural network" in {

    // see: https://iamtrask.github.io/2015/07/12/basic-python-network/

    ns.rand.setSeed(231)

    // 1 on 1 correlation between 1st column of x and outcome y
    val x = ns
      .array( //
          0, 0, 1, //
          1, 1, 1, //
          1, 0, 1, //
          0, 1, 1)
      .reshape(4, 3)
    val y = ns.array(0, 1, 1, 0).T

    val w0 = 2 * ns.rand(3, 4) - 1
    val w1 = 2 * ns.rand(4, 1) - 1

    for (j <- 0 until 10000) {
      val l1 = 1 / (1 + ns.exp(-ns.dot(x, w0)))
      val l2 = 1 / (1 + ns.exp(-ns.dot(l1, w1)))

      val l2_error = ns.mean(ns.abs(y - l2)).squeeze()
      if (j % 1000 == 0) println(s"$j: pred: $l2 error: $l2_error")

      val l2_delta = (y - l2) * (l2 * (1 - l2))
      val l1_delta = l2_delta.dot(w1.T) * (l1 * (1 - l1))
      w1 += l1.T.dot(l2_delta)
      w0 += x.T.dot(l1_delta)
    }

    def predict(x: Tensor): Tensor = {
      val l1 = 1 / (1 + ns.exp(-ns.dot(x, w0)))
      val l2 = 1 / (1 + ns.exp(-ns.dot(l1, w1)))
      l2
    }

    // unseen x's

    val unseen_x = Tensor( //
        0, 0, 0, //
        0, 1, 0, //
        1, 0, 0, //
        1, 1, 0 //
        ).reshape(4, 3)

    val unseen_y = Tensor(0, 0, 1, 1).T

    val p = ns.round(predict(unseen_x))

    assert(ns.all(p == unseen_y))

  }

}
