
Numsca: Numpy for Scala
=========================

[![Maven Central](https://maven-badges.herokuapp.com/maven-central/be.botkop/numsca_2.12/badge.svg)](https://maven-badges.herokuapp.com/maven-central/be.botkop/numsca_2.12)
[![Build Status](https://travis-ci.org/botkop/numsca.svg?branch=master)](https://travis-ci.org/botkop/numsca)

Numsca is Numpy for Scala.

Here's the famous [neural network in 11 lines of Python](http://iamtrask.github.io/2015/07/12/basic-python-network/), translated to Numsca:

```scala
import botkop.{numsca => ns}
val x = ns.array(0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1).reshape(4, 3)
val y = ns.array(0, 1, 1, 0).T
val w0 = 2 * ns.rand(3, 4) - 1
val w1 = 2 * ns.rand(4, 1) - 1
for (j <- 0 until 60000) {
  val l1 = 1 / (1 + ns.exp(-ns.dot(x, w0)))
  val l2 = 1 / (1 + ns.exp(-ns.dot(l1, w1)))
  val l2Delta = (y - l2) * (l2 * (1 - l2))
  val l1Delta = l2Delta.dot(w1.T) * (l1 * (1 - l1))
  w1 += l1.T.dot(l2Delta)
  w0 += x.T.dot(l1Delta)
}
``` 

Another example: a Scala translation of Andrej Karpathy's 
['Minimal character-level language model with a Vanilla Recurrent Neural Network'](src/main/scala/botkop/numsca/samples/MinCharRnn.scala).
(Compare with Andrej Karpathy's original [post](https://gist.github.com/karpathy/d4dee566867f8291f086).)

Also have a look at [Scorch](https://github.com/botkop/scorch), a neural net framework in the spirit of [PyTorch](http://pytorch.org/), which uses Numsca.


## Why?
I love Scala. I teach myself deep learning. Everything in deep learning is written in Python. 
This library helps me to quickly translate Python and Numpy code to my favorite language. 

I hope you find it useful. 

Pull requests welcome.

## Disclaimer
This is far from an exhaustive copy of Numpy's functionality. I'm adding functionality as I go. 
That being said, I think many of the most interesting aspects of Numpy like slicing, broadcasting and indexing 
have been successfully implemented.

## Under the hood
Numsca piggybacks on [Nd4j](https://nd4j.org/). Thanks, people!

## Dependency
Add this to build.sbt:
```scala
libraryDependencies += "be.botkop" %% "numsca" % "0.1.4"
```

## Importing Numsca
```scala
import botkop.{numsca => ns}
import ns.Tensor
```

## Creating a Tensor

```scala
scala> Tensor(3, 2, 1, 0)
[3.00,  2.00,  1.00,  0.00]

scala> ns.zeros(3, 3)
[[0.00,  0.00,  0.00],
 [0.00,  0.00,  0.00],
 [0.00,  0.00,  0.00]]

scala> ns.ones(3, 2)
[[1.00,  1.00],
 [1.00,  1.00],
 [1.00,  1.00]]
 
scala> val ta: Tensor = ns.arange(10)
[0.00,  1.00,  2.00,  3.00,  4.00,  5.00,  6.00,  7.00,  8.00,  9.00]

scala> val tb: Tensor = ns.reshape(ns.arange(9), 3, 3)
[[0.00,  1.00,  2.00],
 [3.00,  4.00,  5.00],
 [6.00,  7.00,  8.00]]
 
 scala> val tc: Tensor = ns.reshape(ns.arange(2 * 3 * 4), 2, 3, 4)
 [[[0.00,  1.00,  2.00,  3.00],
   [4.00,  5.00,  6.00,  7.00],
   [8.00,  9.00,  10.00,  11.00]],
 
  [[12.00,  13.00,  14.00,  15.00],
   [16.00,  17.00,  18.00,  19.00],
   [20.00,  21.00,  22.00,  23.00]]]
```

## Access
Single element
```scala
scala> ta(0)
res10: botkop.numsca.Tensor = 0.00

scala> tc(0, 1, 2)
res14: botkop.numsca.Tensor = 6.00
```
Get the value of a single element Tensor:
```scala
scala> ta(0).squeeze()
res11: Double = 0.0
```
Slice
```scala
scala> tc(0)
res7: botkop.numsca.Tensor =
[[0.00,  1.00,  2.00,  3.00],
 [4.00,  5.00,  6.00,  7.00],
 [8.00,  9.00,  10.00,  11.00]]
 
scala> tc(0, 1)
res8: botkop.numsca.Tensor = [4.00,  5.00,  6.00,  7.00]
```

## Update
In place
```scala
scala> val t = ta.copy()
t: botkop.numsca.Tensor = [0.00,  1.00,  2.00,  3.00,  4.00,  5.00,  6.00,  7.00,  8.00,  9.00]

scala> t(3) := -5
scala> t
res16: botkop.numsca.Tensor = [0.00,  1.00,  2.00,  -5.00,  4.00,  5.00,  6.00,  7.00,  8.00,  9.00]

scala> t(0) += 7
scala> t
res18: botkop.numsca.Tensor = [7.00,  1.00,  2.00,  -5.00,  4.00,  5.00,  6.00,  7.00,  8.00,  9.00]
```

Array wise
```scala
scala> val a2 = 2 * ta
val a2 = 2 * ta
a2: botkop.numsca.Tensor = [0.00,  2.00,  4.00,  6.00,  8.00,  10.00,  12.00,  14.00,  16.00,  18.00]
```

## Slicing
Note: 
- negative indexing is supported
- Python notation ```t[:3]``` must be written as ```t(0 :> 3)``` or ```t(:>(3))``` 

Not supported (yet):
- step size
- ellipsis

### Single dimension
#### Slice over a single dimension

```scala
scala> val a0 = ta.copy().reshape(10, 1)
a0: botkop.numsca.Tensor = [0.00,  1.00,  2.00,  3.00,  4.00,  5.00,  6.00,  7.00,  8.00,  9.00]

scala> val a1 = a0(1 :>)
a1: botkop.numsca.Tensor = [1.00,  2.00,  3.00,  4.00,  5.00,  6.00,  7.00,  8.00,  9.00]

scala> val a2 = a0(0 :> -1)
a2: botkop.numsca.Tensor = [0.00,  1.00,  2.00,  3.00,  4.00,  5.00,  6.00,  7.00,  8.00]

scala> val a3 = a1 - a2
a3: botkop.numsca.Tensor = [1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00]

scala> ta(:>, 5 :>)
res19: botkop.numsca.Tensor = [5.00,  6.00,  7.00,  8.00,  9.00]

scala> ta(:>, -3 :>)
res4: botkop.numsca.Tensor = [7.00,  8.00,  9.00]
```

#### Update single dimension slice

```scala
scala> val t = ta.copy()
t: botkop.numsca.Tensor = [0.00,  1.00,  2.00,  3.00,  4.00,  5.00,  6.00,  7.00,  8.00,  9.00]
```
Assign another tensor
```scala
scala> t(2 :> 5) := -ns.ones(3)
scala> t
res6: botkop.numsca.Tensor = [0.00,  1.00,  -1.00,  -1.00,  -1.00,  5.00,  6.00,  7.00,  8.00,  9.00]
```
Assign a value
```scala
scala> t(2 :> 5) := 33
scala> t
res8: botkop.numsca.Tensor = [0.00,  1.00,  33.00,  33.00,  33.00,  5.00,  6.00,  7.00,  8.00,  9.00]
```
Update in place
```scala
scala> t(2 :> 5) -= 1
scala> t
res10: botkop.numsca.Tensor = [0.00,  1.00,  32.00,  32.00,  32.00,  5.00,  6.00,  7.00,  8.00,  9.00]

```

### Multidimensional slices
```scala
scala> tb
res11: botkop.numsca.Tensor =
[[0.00,  1.00,  2.00],
 [3.00,  4.00,  5.00],
 [6.00,  7.00,  8.00]]
 
scala> tb(2:>, :>)
res15: botkop.numsca.Tensor = [6.00,  7.00,  8.00]
```
Mixed range/integer indexing. Note that integers are implicitly translated to ranges, 
and this differs from Python. 
```scala
scala> tb(1, 0 :> -1)
res1: botkop.numsca.Tensor = [3.00,  4.00]
```

## Fancy indexing
### Boolean indexing
```scala
scala> val c = ta < 5 && ta > 1
c: botkop.numsca.Tensor = [0.00,  0.00,  1.00,  1.00,  1.00,  0.00,  0.00,  0.00,  0.00,  0.00]
```
This returns a TensorSelection:
```scala
scala> val d = ta(c)
d: botkop.numsca.TensorSelection = TensorSelection([0.00,  1.00,  2.00,  3.00,  4.00,  5.00,  6.00,  7.00,  8.00,  9.00],[[I@153ea1aa,None)
```
Which is implicitly converted to a Tensor when needed:
```scala
scala> val d: Tensor = ta(c)
d: botkop.numsca.Tensor = [2.00,  3.00,  4.00]
```
Or you can force it to become a Tensor:
```scala
scala> ta(c).asTensor
res10: botkop.numsca.Tensor = [2.00,  3.00,  4.00]
```
Updating:
```scala
scala> val t = ta.copy()
scala> t(ta < 5 && ta > 1) := -7
res6: botkop.numsca.Tensor = [0.00,  1.00,  -7.00,  -7.00,  -7.00,  5.00,  6.00,  7.00,  8.00,  9.00]
```
Selection over multiple dimensions:
```scala
scala> val c: Tensor = tc(tc % 5 == 0)
c: botkop.numsca.Tensor = [0.00,  5.00,  10.00,  15.00,  20.00]
```
Updating over multiple dimensions:
```scala
scala> val t1 = tc.copy()
t1: botkop.numsca.Tensor =
[[[0.00,  1.00,  2.00,  3.00],
  [4.00,  5.00,  6.00,  7.00],
  [8.00,  9.00,  10.00,  11.00]],

 [[12.00,  13.00,  14.00,  15.00],
  [16.00,  17.00,  18.00,  19.00],
  [20.00,  21.00,  22.00,  23.00]]]
  
scala> t1(t1 > 5 && t1 < 15) *= 2
res21: botkop.numsca.Tensor =
[[[0.00,  1.00,  2.00,  3.00],
  [4.00,  5.00,  12.00,  14.00],
  [16.00,  18.00,  20.00,  22.00]],

 [[24.00,  26.00,  28.00,  15.00],
  [16.00,  17.00,  18.00,  19.00],
  [20.00,  21.00,  22.00,  23.00]]]
```
### List of location indexing
```scala
scala> val primes = Tensor(2, 3, 5, 7, 11, 13, 17, 19, 23)

scala> val idx = Tensor(3, 4, 1, 2, 2)

scala> primes(idx).asTensor
res23: botkop.numsca.Tensor = [7.00,  11.00,  3.00,  5.00,  5.00]

```
Reshape according to index:
```scala
scala> tb
res25: botkop.numsca.Tensor =
[[0.00,  1.00,  2.00],
 [3.00,  4.00,  5.00],
 [6.00,  7.00,  8.00]]

scala> primes(tb).asTensor
res24: botkop.numsca.Tensor =
[[2.00,  3.00,  5.00],
 [7.00,  11.00,  13.00],
 [17.00,  19.00,  23.00]]
```
Use as a look-up table:
```scala
scala> val numSamples = 4
       val numClasses = 3
       val x = ns.arange(numSamples * numClasses).reshape(numSamples, numClasses)
       val y = Tensor(0, 1, 2, 1)
       val z: Tensor = x(ns.arange(numSamples), y)
res26: botkop.numsca.Tensor = [0.00,  4.00,  8.00,  10.00]
```
Update along a single dimension:
```scala
scala> val primes = Tensor(2, 3, 5, 7, 11, 13, 17, 19, 23)
primes: botkop.numsca.Tensor = [2.00,  3.00,  5.00,  7.00,  11.00,  13.00,  17.00,  19.00,  23.00]

scala> val idx = Tensor(3, 4, 1, 2, 2)
idx: botkop.numsca.Tensor = [3.00,  4.00,  1.00,  2.00,  2.00]

scala> primes(idx) := 0

scala> primes
res1: botkop.numsca.Tensor = [2.00,  0.00,  0.00,  0.00,  0.00,  13.00,  17.00,  19.00,  23.00]
```
Multiple dimensions
```scala

scala> val a = ns.arange(6).reshape(3, 2) + 1
a: botkop.numsca.Tensor =
[[1.00,  2.00],
 [3.00,  4.00],
 [5.00,  6.00]]

scala> val s1 = Tensor(0, 1, 2)
s1: botkop.numsca.Tensor = [0.00,  1.00,  2.00]

scala> val s2 = Tensor(0, 1, 0)
s2: botkop.numsca.Tensor = [0.00,  1.00,  0.00]

scala> val r1: Tensor = a(s1, s2)
r1: botkop.numsca.Tensor = [1.00,  4.00,  5.00]
```
An index will be broadcast if needed:
```scala
scala> val y = ns.arange(35).reshape(5, 7)
y: botkop.numsca.Tensor =
[[0.00,  1.00,  2.00,  3.00,  4.00,  5.00,  6.00],
 [7.00,  8.00,  9.00,  10.00,  11.00,  12.00,  13.00],
 [14.00,  15.00,  16.00,  17.00,  18.00,  19.00,  20.00],
 [21.00,  22.00,  23.00,  24.00,  25.00,  26.00,  27.00],
 [28.00,  29.00,  30.00,  31.00,  32.00,  33.00,  34.00]]

scala> val r5: Tensor = y(Tensor(0, 2, 4), Tensor(1))
r5: botkop.numsca.Tensor = [1.00,  15.00,  29.00]
```

Update along multiple dimensions:
```scala
scala> val a = ns.arange(6).reshape(3, 2) + 1
a: botkop.numsca.Tensor =
[[1.00,  2.00],
 [3.00,  4.00],
 [5.00,  6.00]]

scala> val s1 = Tensor(1, 1, 2)
s1: botkop.numsca.Tensor = [1.00,  1.00,  2.00]

scala> val s2 = Tensor(0, 1, 0)
s2: botkop.numsca.Tensor = [0.00,  1.00,  0.00]

scala> a(s1, s2) := 0
res1: botkop.numsca.Tensor =
[[1.00,  2.00],
 [0.00,  0.00],
 [0.00,  6.00]]
```
## Broadcasting

```scala
scala> val x = ns.arange(4)
x: botkop.numsca.Tensor = [0.00,  1.00,  2.00,  3.00]

scala> val xx = x.reshape(4, 1)
xx: botkop.numsca.Tensor = [0.00,  1.00,  2.00,  3.00]

scala> val y = ns.ones(5)
y: botkop.numsca.Tensor = [1.00,  1.00,  1.00,  1.00,  1.00]

scala> val z = ns.ones(3, 4)
    val z = ns.ones(3, 4)
[[1.00,  1.00,  1.00,  1.00],
 [1.00,  1.00,  1.00,  1.00],
 [1.00,  1.00,  1.00,  1.00]]

scala> (xx + y)
[[1.00,  1.00,  1.00,  1.00,  1.00],
 [2.00,  2.00,  2.00,  2.00,  2.00],
 [3.00,  3.00,  3.00,  3.00,  3.00],
 [4.00,  4.00,  4.00,  4.00,  4.00]]

scala> x + z
[[1.00,  2.00,  3.00,  4.00],
 [1.00,  2.00,  3.00,  4.00],
 [1.00,  2.00,  3.00,  4.00]]
```
Outer sum:
```scala
scala> val a = Tensor(0.0, 10.0, 20.0, 30.0).reshape(4, 1)
a: botkop.numsca.Tensor = [0.00,  10.00,  20.00,  30.00]

scala> val b = Tensor(1.0, 2.0, 3.0)
b: botkop.numsca.Tensor = [1.00,  2.00,  3.00]

scala> a + b
res6: botkop.numsca.Tensor =
[[1.00,  2.00,  3.00],
 [11.00,  12.00,  13.00],
 [21.00,  22.00,  23.00],
 [31.00,  32.00,  33.00]]

```

Vector Quantization from [EricsBroadcastingDoc](http://scipy.github.io/old-wiki/pages/EricsBroadcastingDoc):
```scala
scala> val observation = Tensor(111.0, 188.0)

scala> val codes = Tensor( 102.0, 203.0, 132.0, 193.0, 45.0, 155.0, 57.0, 173.0).reshape(4, 2)
codes: botkop.numsca.Tensor =
[[102.00,  203.00],
 [132.00,  193.00],
 [45.00,  155.00],
 [57.00,  173.00]]

scala> val diff = codes - observation
diff: botkop.numsca.Tensor =
[[-9.00,  15.00],
 [21.00,  5.00],
 [-66.00,  -33.00],
 [-54.00,  -15.00]]

scala> val dist = ns.sqrt(ns.sum(ns.square(diff), axis = -1))
dist: botkop.numsca.Tensor = [17.49,  21.59,  73.79,  56.04]

scala>     val nearest = ns.argmin(dist).squeeze()
nearest: Double = 0.0

```
